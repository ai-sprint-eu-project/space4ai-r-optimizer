/*
Copyright 2021 AI-SPRINT

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/**
* \file UtilizationHeuristic.cpp
*
* \brief Defines the methods of the UtilizationHeuristic class
*
* \author Federica Filippini
*/

#include "src/Algorithms/UtilizationHeuristic.hpp"

namespace Space4AI
{

UtilizationHeuristic::UtilizationHeuristic(const System& system_):
UtilizationHeuristic(system_, "fixed", SelectedResources())
{}

UtilizationHeuristic::UtilizationHeuristic(
  const System& system_, const std::string& rule_
):
UtilizationHeuristic(system_, rule_, SelectedResources())
{}

UtilizationHeuristic::UtilizationHeuristic(
  const System& system_, 
  const std::string& rule_, 
  const SelectedResources& fixed_edge_and_curr_rt_vms_
):
system(system_), rule(rule_), fixed_edge_and_curr_rt_vms(fixed_edge_and_curr_rt_vms_)
{
  const auto& all_resources = system.get_system_data().get_all_resources();
  local_info.modified_res.resize(ResIdxFromType(ResourceType::Count));

  for(size_t i = 0; i < local_info.modified_res.size(); ++i)
  {
    local_info.modified_res[i].resize(all_resources.get_number_resources(i));
  }
}

EliteResult
UtilizationHeuristic::run(
  const Solution& initial_solution,
  double min_utilization,
  double max_utilization
)
{
  Logger::Info("Starting UtilizationHeuristic algorithm");

  // initialize the containers to store the solution
  const size_t num_top_sols = 1;
  EliteResult elite(num_top_sols);
  EliteResult unfeasible(1);
  Logger::Info(
    "Elite container initialized with " + std::to_string(num_top_sols) + " spaces"
  );

  // initialize LocalInfo object to store changes
  local_info.reset();
  local_info.old_local_parts_perfs_ptr = initial_solution.get_local_parts_perfs();
  local_info.old_local_parts_delays_ptr = initial_solution.get_local_parts_delays();

  // determine the new solution
  Solution sol(
    update_solution(initial_solution, min_utilization, max_utilization)
  );
  sol.set_selected_resources(system);

  // check feasibility
  if(sol.check_feasibility(system))
  {
    // evaluate the objective function
    sol.objective_function(system);

    // if feasible, add to the elite set
    Logger::Debug("run: the solution is feasible");
    elite.add(std::move(sol));
  }  
  else
  {
    // otherwise, save as unfeasible solution
    Logger::Debug("run: the solution is NOT feasible");
    unfeasible.add(std::move(sol));
  }

  Logger::Info("Finished UtilizationHeuristic algorithm");
  Logger::Info(
    "Number of top feasible solutions found: " + std::to_string(elite.get_size())
  );

  // if no feasible solutions are found, return the unfeasible one
  if (elite.get_size() == 0)
    elite.swap(unfeasible);

  return elite;
}

size_t
UtilizationHeuristic::compute_new_number(
  double utilization, 
  size_t number, 
  double min_utilization, 
  double max_utilization
) const
{
  size_t new_number = number;

  if (utilization > max_utilization)
  {
    if (rule == "fixed")
      new_number += fixed_incr;
    else if (rule == "percentage")
      new_number = ceil(new_number * (1 + incr_percentage));
  }
  else if (utilization < min_utilization)
  {
    if (rule == "fixed")
      new_number -= fixed_decr;
    else if (rule == "percentage")
      new_number = ceil(new_number * (1 - decr_percentage));
  }

  return new_number;
}

size_t
UtilizationHeuristic::get_max_number(
  ResourceType res_type, size_t res_idx
) const
{
  size_t max_number = 0;

  size_t res_type_idx = ResIdxFromType(res_type);
  if (res_type_idx == ResIdxFromType(ResourceType::Edge))
  {
    const auto& selected = fixed_edge_and_curr_rt_vms.get_selected_edge();
    max_number = selected[res_idx];
  }
  else if (res_type_idx == ResIdxFromType(ResourceType::VM))
  {
    const auto& selected = fixed_edge_and_curr_rt_vms.get_selected_vms();
    max_number = selected[res_idx];
  }
  else
  {
    max_number = 1;
  }

  if (max_number == 0)
  {
    const auto& all_resources = system.get_system_data().get_all_resources();
    max_number = all_resources.get_number_avail(res_type, res_idx);
  }

  return max_number;
}

Solution
UtilizationHeuristic::update_solution(
  const Solution& initial_solution,
  double min_utilization, 
  double max_utilization
)
{
  Logger::Debug("update_solution: starting new update ...");

  // get initial solution data
  const auto& i_solution_data = initial_solution.get_solution_data();
  const auto& i_y_hat = initial_solution.get_y_hat();
  const auto& i_used_resources = initial_solution.get_used_resources();
  const auto& i_n_used_resources = initial_solution.get_n_used_resources();

  // initialize new solution
  Solution solution(initial_solution);

  // initialize container for visited resources
  size_t res_type_idx_count = ResIdxFromType(ResourceType::Count);
  std::vector<std::vector<size_t>> visited_res(res_type_idx_count);
  for (size_t type_idx = 0; type_idx < res_type_idx_count; ++type_idx)
    visited_res[type_idx].resize(i_n_used_resources[type_idx].size(), false);

  // get system data and description
  const auto& system_data = system.get_system_data();
  const auto& components = system_data.get_components();
  const auto& performance = system.get_performance();

  // loop over components
  for(std::size_t i = 0; i < components.size(); ++i)
  {
    // loop over initial used resources
    for(size_t j = 0; j < i_used_resources[i].size(); ++j)
    {
      // get idxs
      const auto& tuple = i_used_resources[i][j];
      const auto& part_idx = std::get<0>(tuple);
      const auto& res_type_idx = std::get<1>(tuple);
      const auto& res_idx = std::get<2>(tuple);

      // check if the resource was already visited
      if(!visited_res[res_type_idx][res_idx])
      {
        // if the resource is an Edge or Cloud VM
        if(res_type_idx != ResIdxFromType(ResourceType::Faas))
        {
          // get number
          const auto& number = i_y_hat[i][res_type_idx][part_idx][res_idx];

          // if the performance model supports co-location...
          const auto& model = performance[i][res_type_idx][part_idx][res_idx];
          if (model->get_allows_colocation())
          {
            // ...compute utilization
            double utilization = model->compute_utilization(
              ResTypeFromIdx(res_type_idx), res_idx, system_data, i_solution_data
            );
            Logger::Debug(
              "update_solution: utilization of resource " + 
              std::to_string(res_idx) + 
              " of type " + 
              std::to_string(res_type_idx) + 
              " is: " +
              std::to_string(utilization)
            );

            // compute the new number of instances to consider
            size_t new_n = compute_new_number(
              utilization, number, min_utilization, max_utilization
            );
            Logger::Debug(
              "update_solution: assigning " +
              std::to_string(new_n) +
              " instances (previous number: " +
              std::to_string(number) +
              ")"
            );

            // compute the maximum number of instances
            size_t max_number = get_max_number(
              ResTypeFromIdx(res_type_idx), 
              res_idx
            );

            Logger::Trace(
              "update_solution: the number of available instances is " +
              std::to_string(max_number)
            );

            // update the number
            if (new_n != number && new_n > 0 && new_n <= max_number)
            {
              solution.set_instance_number(
                ResTypeFromIdx(res_type_idx), 
                res_idx,
                new_n
              );
              Logger::Trace("update_solution: new number assigned");

              // record change in local_info
              local_info.active = true;
              local_info.modified_res[res_type_idx][res_idx] = true;
            }
          }
        }
      }

      // set the resource as visited
      visited_res[res_type_idx][res_idx] = true;
    }
  }
  
  Logger::Debug("update_solution: Done!");
  
  return solution;
}

} //namespace Space4AI
