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
* \file UtilizationHeuristic.hpp
*
* \brief Defines the UtilizationHeuristic algorithm to solve the
*        optimization problem
*
* \author Federica Filippini
*/

#ifndef UTILIZATION_HEURISTIC_HPP_
#define UTILIZATION_HEURISTIC_HPP_

#include "src/System/System.hpp"
#include "src/Solution/Solution.hpp"
#include "src/Solution/EliteResult.hpp"

namespace Space4AI
{
/** Class to define the UtilizationHeuristic algorithm to solve the 
 * optimization problem */
class UtilizationHeuristic
{
  public:

    UtilizationHeuristic(const System& system_);

    UtilizationHeuristic(const System& system_, const std::string& rule_);

    UtilizationHeuristic(
      const System& system_, 
      const std::string& rule_, 
      const SelectedResources& fixed_edge_and_curr_rt_vms_
    );

    /** Method to generate the new solution
    *
    *   \param system Object containing all the data structures of the System
    *   \param initial_solution Initial Solution
    *   \param min_utilization Minimum utilization threshold
    *   \param max_utilization Maximum utilization threshold
    *   \return EliteResult class containing num_top_sols solutions ordered by cost
    */
    EliteResult run(
      const Solution& initial_solution,
      double min_utilization,
      double max_utilization
    );

    /** Set variation percentages */
    void set_decr_percentage(double decr_percentage_)
    {decr_percentage = decr_percentage_;}
    void set_incr_percentage(double incr_percentage_)
    {incr_percentage = incr_percentage_;}

    /** Set the current run-time solution selected resources (e.g., 
     * at design-time) on Edge and VM */
    void set_selected_resources(
      const SelectedResources& fixed_edge_and_curr_rt_vms_
    ){fixed_edge_and_curr_rt_vms = fixed_edge_and_curr_rt_vms_;}

  private:

    size_t compute_new_number(
      double utilization, 
      size_t number, 
      double min_utilization, 
      double max_utilization
    ) const;
    
    size_t get_max_number(ResourceType res_type, size_t res_idx) const;

    Solution update_solution(
      const Solution& initial_solution, 
      double min_utilization, 
      double max_utilization
    );

    /** System description */
    const System& system;

    /** Update rule */
    const std::string rule = "fixed";

    /** Fixed increment to update resource */
    size_t fixed_decr = 1;
    size_t fixed_incr = 1;

    /** Increase/Decrease percentages to update resources */
    double decr_percentage = 0.1;
    double incr_percentage = 0.1;

    /** Current run-time solution selected resources (e.g., at design-time) 
     * on Edge and VM */
    SelectedResources fixed_edge_and_curr_rt_vms;

    /** local info to track modifications od Local Search */
    LocalInfo local_info;

};

} // namespace Space4AI

#endif /* UTILIZATION_HEURISTIC_HPP_ */
