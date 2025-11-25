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
* \file SolutionData.cpp
*
* \brief Defines the methods of the class SolutionData.
*
* \author Federica Filippini
*/

#include "src/Solution/SolutionData.hpp"
#include "src/Logger.hpp"

namespace Space4AI
{
void
SolutionData::set_instance_number(
  ResourceType res_type, size_t res_idx, size_t n
)
{
  // resource type index
  size_t res_type_idx = ResIdxFromType(res_type);
  
  // update n_used_resources
  n_used_resources[res_type_idx][res_idx] = n;
  Logger::Trace("  set_instance_number: updated n_used_resources");

  // loop over all components
  for (std::size_t i = 0; i < y_hat.size(); ++i)
  {
    // loop over all partitions
    for (std::size_t h = 0; h < y_hat[i][res_type_idx].size(); ++h)
    {
      // update y_hat
      if (y_hat[i][res_type_idx][h][res_idx] != 0)
      {
        y_hat[i][res_type_idx][h][res_idx] = n;

        Logger::Trace(
          "  set_instance_number: updated y_hat for component " +
          std::to_string(i) +
          ", partition " +
          std::to_string(h)
        );
      }
    }
  }
}
} // namespace Space4AI
