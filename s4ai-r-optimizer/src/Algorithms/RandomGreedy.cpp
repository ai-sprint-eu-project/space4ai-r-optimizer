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
* \file RandomGreedy.cpp.in
*
* \brief Defines the methods of the RandomGreedy class
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#include "src/Algorithms/RandomGreedy.hpp"
#include "src/Algorithms/ParallelConfig.hpp"

namespace Space4AI
{

EliteResult
RandomGreedy::random_greedy(
  const System& system,
  size_t max_it,
  size_t num_top_sols,
  bool reproducibility
)
{
  Logger::Info("Starting Random Greedy DT algorithm with reproducibility flag raised to: " + std::to_string(reproducibility));

  if(reproducibility)
  {
    rng.seed(seed);
  }
  else
  {
    std::random_device dev;
    rng.seed(dev());
  }

  EliteResult elite(num_top_sols);
  Logger::Info("Elite container initialized with " + std::to_string(num_top_sols) + " spaces");

  MY_PRAGMA(omp parallel for default(none) shared(max_it, num_top_sols, elite, system) schedule(dynamic))
    for(size_t it = 0; it < max_it; ++it)
    {
      Logger::Debug("**** iteration: " + std::to_string(it) +  " ****");

      Solution sol(create_random_initial_solution(system));
      if(sol.check_feasibility(system))
      {
        Logger::Debug("random_greedy: the solution is feasible");
        const std::vector<size_t> res_type_idxs = {ResIdxFromType(ResourceType::VM), ResIdxFromType(ResourceType::Edge)};
        const UsedResourcesNumberType& n_used_resources = sol.get_n_used_resources();
        // update the cluster size of edge and VM resources
        // loop on edge and VM types
        for(size_t res_type_idx : res_type_idxs)
        {
          //loop on resources
          for(size_t res_idx = 0; res_idx < n_used_resources[res_type_idx].size(); ++res_idx)
          {
            if(n_used_resources[res_type_idx][res_idx] > 1)
            {
              sol = reduce_cluster_size(sol, res_type_idx, res_idx, system);
            }
          }
        }
        sol.objective_function(system);
        sol.set_selected_resources(system);
        #warning Was it better to compute the cost afterwards?
        MY_PRAGMA(omp critical)
        elite.add(std::move(sol));
      }
    }

  Logger::Info("Finished Random Greedy DT algorithm");
  Logger::Info("Number of top feasible solutions found: " + std::to_string(elite.get_size()));
#ifdef PARALLELIZATION
  elite.set_num_threads(omp_get_max_threads()); // set the current number of used threads;
#endif
  return elite;
}

Solution
RandomGreedy::create_random_initial_solution(
  const System& system
)
{
  Logger::Debug("create_random_initial_solution: starting new iteration ...");

  Solution solution(system);
  const auto& system_data = system.get_system_data();
  const auto& components = system_data.get_components();
  const auto& cls = system_data.get_cls();
  const auto& all_resources = system_data.get_all_resources();
  const auto& comp_name_to_idx = system_data.get_comp_name_to_idx();
  const auto& compatibility_matrix = system_data.get_compatibility_matrix();
  const size_t edge_type_idx = ResIdxFromType(ResourceType::Edge);
  const size_t vm_type_idx = ResIdxFromType(ResourceType::VM);
  const auto faas_type_idx = ResIdxFromType(ResourceType::Faas);
  Logger::Debug("create_random_initial_solution: Initializing and resizing members ...");
  //declare solution members
  YHatType y_hat;
  UsedResourcesOrderedType used_resources; //vettore di vettore ORDINATO di tuple
  UsedResourcesNumberType n_used_resources;
  const size_t comp_num = comp_name_to_idx.size();
  const size_t res_type_idx_count = ResIdxFromType(ResourceType::Count);
  // resize y_hat, used_resources
  y_hat.resize(comp_num);
  used_resources.resize(comp_num);
  //resize n_used_resources
  n_used_resources.resize(2); // Edge, VM (don't like the two ...)
  n_used_resources[edge_type_idx].resize(all_resources.get_number_resources(edge_type_idx)); // Index of Edge must be 0
  n_used_resources[vm_type_idx].resize(all_resources.get_number_resources(vm_type_idx));    // Index of VM must be 1

  //loop on components
  for(size_t i = 0; i < comp_num; ++i)
  {
    y_hat[i].resize(res_type_idx_count);

    //loop on resource types
    for(size_t j = 0; j < res_type_idx_count; ++j)
    {
      y_hat[i][j].resize(components[i].get_partitions().size());

      //loop on partitions
      for(size_t k = 0; k < y_hat[i][j].size(); ++k)
      {
        y_hat[i][j][k].resize(all_resources.get_number_resources(j));
      }
    }
  }

  std::vector<std::vector<bool>> candidate_resources(res_type_idx_count); //per ogni tipo di risorsa, per ogni risorsa, si o no

  // initialize candidate resources
  for(size_t i = 0; i < res_type_idx_count; ++i)
  {
    const size_t n_res = all_resources.get_number_resources(i);

    if(i == ResIdxFromType(ResourceType::Faas))
    {
      candidate_resources[i].resize(n_res, true); // select all the faas as candidates
    }
    else
    {
      candidate_resources[i].resize(n_res, false);
    }
  }

  // Selecting candidate_resources (Edge and VM, Faas already selected all)
  Logger::Debug("create_random_initial_solution: Selecting candidate resources for Edge and VM...");
  // Getting the initial deployment edge and curr runtime selected vms
  const auto& selected_edge = fixed_edge_and_curr_rt_vms.get_selected_edge();
  const auto& selected_vms_by_cl = fixed_edge_and_curr_rt_vms.get_selected_vms_by_cl();

  // EDGE RESOURCES
  if(selected_edge.size() > 0)  // If I provided old selected resources
  {
    std::copy(
      selected_edge.begin(), selected_edge.end(), candidate_resources[edge_type_idx].begin());
  }
  else // randomly select a resource for each cls
  {
    for(size_t cl_idx = 0; cl_idx < cls[edge_type_idx].size(); ++cl_idx)
    {
      //pick a random resource of the cl
      const std::vector<size_t> res_idxs = cls[edge_type_idx][cl_idx].get_res_idxs();
      const size_t n_res = res_idxs.size();
      std::uniform_int_distribution<decltype(rng)::result_type> dist(0, n_res - 1);
      const size_t random_res_idx = res_idxs[dist(rng)];
      candidate_resources[edge_type_idx][random_res_idx] = true;
      Logger::Trace(
        "create_random_initial_solution: selecting Random Candidate resource..."
        "resource_type: " + std::to_string(edge_type_idx) + " res_idx: " + std::to_string(random_res_idx));
    }
  }

  // VM RESOURCES
  for(size_t cl_idx = 0; cl_idx < cls[vm_type_idx].size(); ++cl_idx)
  {
    if(selected_vms_by_cl.size() > 0 && selected_vms_by_cl[cl_idx].first)
    {
      candidate_resources[vm_type_idx][selected_vms_by_cl[cl_idx].second] = true;
    }
    else
    {
      //pick a random resource of the cl
      const std::vector<size_t> res_idxs = cls[vm_type_idx][cl_idx].get_res_idxs();
      const size_t n_res = res_idxs.size();
      std::uniform_int_distribution<decltype(rng)::result_type> dist(0, n_res - 1);
      const size_t random_res_idx = res_idxs[dist(rng)];
      candidate_resources[vm_type_idx][random_res_idx] = true;
      Logger::Trace(
        "create_random_initial_solution: selecting Random Candidate resource..."
        "resource_type: " + std::to_string(vm_type_idx) + " res_idx: " + std::to_string(random_res_idx));
    }
  }

  //Assign components
  //loop over components
  Logger::Debug("create_random_initial_solution: Assigning the components...");

  for(size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx)
  {
    Logger::Trace("create_random_initial_solution:  ***** Component: " + std::to_string(comp_idx) + " ******");
    //randomly select a deployment for that component
    const auto& deployments = components[comp_idx].get_deployments();
    const size_t n_dep = deployments.size();
    std::uniform_int_distribution<decltype(rng)::result_type> distr(0, n_dep - 1);
    const size_t random_dep_idx = distr(rng);
    const auto& random_dep = deployments[random_dep_idx];
    Logger::Trace("create_random_initial_solution: Selected deployment: " + std::to_string(random_dep_idx));

    //loop over all partitions in the deployment
    for(size_t part_idx : random_dep.get_partition_indices())
    {
      Logger::Trace("create_random_initial_solution: Partition: " + std::to_string(part_idx));
      //list of the resources in the intersection between candidate and compatible resources.
      //pair.first= resource type, pair.second= resource idx
      std::vector<std::pair<size_t, size_t>> resources_instersection;
      resources_instersection.reserve(compatibility_matrix[comp_idx][faas_type_idx][0].size());

      //get the indices of compatible resources and compute the
      //intersection with the selected resources in each computational layer
      // Edge and VM and Faas
      for(size_t res_type_idx = 0; res_type_idx < res_type_idx_count; ++res_type_idx) // Edge or VM
      {
        //loop over the resources of type j
        for(size_t res_idx = 0; res_idx < candidate_resources[res_type_idx].size(); ++res_idx)
        {
          //compute the intersection
          if(candidate_resources[res_type_idx][res_idx] && compatibility_matrix[comp_idx][res_type_idx][part_idx][res_idx])
          {
            resources_instersection.emplace_back(res_type_idx, res_idx);
            Logger::Trace("create_random_initial_solution: added to the intersection resource " + std::to_string(res_idx) + " of type " + std::to_string(res_type_idx));
          }
        }
      }

      // OBS: resources_instersection is NEVER EMPTY! Indeed, the computational layers
      // are built such in a way that if a select one random resource from each layer
      // resources_instersection is never empty
      // THAT'S WHY I DO NOT HAVE TO CHECK if n_inter_res = 0 (~)
      const size_t n_inter_res = resources_instersection.size();
      std::uniform_int_distribution<decltype(rng)::result_type> dist(0, n_inter_res - 1);
      const size_t random_idx = dist(rng);
      const auto& random_resource = resources_instersection[random_idx];
      //update used resources and y_hat
      used_resources[comp_idx].emplace_back(part_idx, random_resource.first, random_resource.second);
      y_hat[comp_idx][random_resource.first][part_idx][random_resource.second] = 1;
      Logger::Trace("create_random_initial_solution: Updated y, y_hat and used_resources");
    }
  }

  std::vector<std::vector<bool>> already_updated_cluster_size(2);
  already_updated_cluster_size[edge_type_idx].resize(all_resources.get_number_resources(edge_type_idx), false);
  already_updated_cluster_size[vm_type_idx].resize(all_resources.get_number_resources(vm_type_idx), false);
  //loop over edge and VM types
  Logger::Debug("create_random_initial_solution: Selecting number of edge and VM resources...");

  for(size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx)
  {
    for(auto [part_idx, res_type_idx, res_idx] : used_resources[comp_idx])
    {
      if(res_type_idx == edge_type_idx || res_type_idx == vm_type_idx)
      {
        Logger::Trace("create_random_initial_solution: resource of type: " + std::to_string(res_type_idx) + " resource index: " + std::to_string(res_idx));

        if(already_updated_cluster_size[res_type_idx][res_idx])
        {
          y_hat[comp_idx][res_type_idx][part_idx][res_idx] = n_used_resources[res_type_idx][res_idx];
          Logger::Trace("create_random_initial_solution: Updated number of resources of comp " + std::to_string(comp_idx) + \
            " part " + std::to_string(part_idx) + " to " + std::to_string(n_used_resources[res_type_idx][res_idx]));
        }
        else
        {
          already_updated_cluster_size[res_type_idx][res_idx] = true;
          size_t number_avail;
          if(res_type_idx == edge_type_idx && selected_edge.size() > 0) // if we are at RT, number avail coincide with the selected resources at edge at initial deployment
            number_avail = selected_edge[res_idx];
          else
            number_avail = all_resources.get_number_avail(ResTypeFromIdx(res_type_idx), res_idx);
          #warning DECIDE HOW TO SET random_number here ...
          // std::uniform_int_distribution<decltype(rng)::result_type> dist(1, number_avail);
          // const size_t random_number = dist(rng);
          const size_t random_number = number_avail;
          y_hat[comp_idx][res_type_idx][part_idx][res_idx] = random_number;
          n_used_resources[res_type_idx][res_idx] = random_number;
          Logger::Trace("create_random_initial_solution: Updated number of resources of comp " + std::to_string(comp_idx) + \
            " part " + std::to_string(part_idx) + " to " + std::to_string(random_number));
        }
      }
    }
  }

  Logger::Debug("create_random_initial_solution: Initializing new random solution...");
  solution.set_y_hat(std::move(y_hat));
  solution.set_used_resources(std::move(used_resources));
  solution.set_n_used_resources(std::move(n_used_resources));
  Logger::Debug("create_random_initial_solution: Done!");
  return solution;
}

Solution
RandomGreedy::reduce_cluster_size(
  const Solution& solution,
  const size_t res_type_idx,
  const size_t res_idx,
  const System& system
)
{
  const auto& used_resources = solution.get_used_resources();
  Solution old_sol = solution;
  Solution new_sol = solution;
  auto y_hat_new = new_sol.get_y_hat();
  auto n_used_resources_new = new_sol.get_n_used_resources();
  bool feasible = true;
  Logger::Debug("reduce_cluster_size: Reducing cluster size...");

  do
  {
    n_used_resources_new[res_type_idx][res_idx]--;

    for(size_t comp_idx = 0; comp_idx < used_resources.size(); ++comp_idx)
    {
      Logger::Trace("reduce_cluster_size: currently on component: " + std::to_string(comp_idx));

      for(auto [p_idx, r_type_idx, r_idx] : used_resources[comp_idx])
      {
        if(res_type_idx == r_type_idx && res_idx == r_idx)
        {
          y_hat_new[comp_idx][r_type_idx][p_idx][r_idx]--;
        }
      }
    }

    new_sol.set_y_hat(y_hat_new);
    new_sol.set_n_used_resources(n_used_resources_new);
    feasible = new_sol.check_feasibility(system);

    if(feasible)
    {
      Logger::Debug("reduce_cluster_size: The solution is still feasible");
      old_sol = new_sol;
    }
    else
    {
      Logger::Debug("reduce_cluster_size: The solution is not feasible anymore reducing cluster size of type: " + std::to_string(res_type_idx) + " and idx " + std::to_string(res_idx));
    }
  }
  while(feasible && (n_used_resources_new[res_type_idx][res_idx] > 1));

  Logger::Debug("reduce_cluster_size: Done reducing cluster size!");

  return old_sol;
}

} //namespace Space4AI
