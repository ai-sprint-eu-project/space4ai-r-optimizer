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
* \file System.cpp
*
* \brief Define the methods of the class System.
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#include "src/System/System.hpp"
#include "src/Solution/Solution.hpp"

namespace Space4AI
{
void
System::read_configuration_file(
  const std::string& system_file, LoadType lambda_, double energy_cost_pct_
)
{
  std::ifstream file(system_file);
  nl::json configuration_file;

  if(!file)
  {
    std::string err_message = "Cannot open: " + system_file + " json file";
    Logger::Error(err_message.c_str());
    throw std::runtime_error(err_message);
  }
  else
  {
    Logger::Info(
      "****** READING CONFIGURATION FILE: " + system_file + " ... ******"
    );
    file >> configuration_file;
  }

  Logger::Info("****** SYSTEM DATA ... ******");
  this->system_data.read_json(configuration_file, lambda_, energy_cost_pct_);
  Logger::Info("********** DONE! **********");

  if(configuration_file.contains("Performance"))
  {
    Logger::Info("****** READING PERFORMANCE MODELS... ******");
    this->initialize_performance(configuration_file.at("Performance"));
    Logger::Info("********** DONE! **********");
  }
  else
  {
    Logger::Error(
      "*System::read_configuration_file(...)*: Performance field \
      (or DemandMatrix field, for old configuration) not present in json file"
    );
    throw std::invalid_argument(
      "*System::read_configuration_file(...)*: Performance field \
      (or DemandMatrix field, for old configuration) not present in json file"
    );
  }
}

void
System::initialize_performance(const nl::json& performance_json)
{
  // initialize demand matrix
  DemandEdgeVMType all_demands;
  all_demands.reserve(this->system_data.components.size());

  // inizialize matrix of average execution times
  MeanTimeType job_mean_times;
  job_mean_times.reserve(this->system_data.components.size());

  // loop over all components
  for(const auto& [idx, comp] : system_data.idx_to_comp_name)
  {
    const auto& comp_data = performance_json.at(comp);
    size_t c_idx = system_data.comp_name_to_idx[comp];
    const auto& partitions = system_data.components[c_idx].get_partitions();
    
    // create and initialize demands_edge_vm_temp (only needed for Edge and VM)
    DemandEdgeVMType::value_type demands_edge_vm_temp(2);
    for(size_t i = 0; i < demands_edge_vm_temp.size(); ++i)
    {
      demands_edge_vm_temp[i].resize(partitions.size());

      for(auto& res_vec : demands_edge_vm_temp[i])
        res_vec.resize(
          system_data.all_resources.get_number_resources(i),
          std::numeric_limits<TimeType>::quiet_NaN()
        );
    }

    // create and initialize perf_temp and mean_time_temp
    PerformanceType::value_type perf_temp(
      ResIdxFromType(ResourceType::Count)
    );
    MeanTimeType::value_type mean_time_temp(
      ResIdxFromType(ResourceType::Count)
    );

    for(std::size_t i = 0; i < perf_temp.size(); ++i)
    {
      perf_temp[i].resize(partitions.size());
      mean_time_temp[i].resize(partitions.size());

      for(std::size_t j = 0; j < perf_temp[i].size(); ++j)
      {
        auto& res_vec = perf_temp[i][j];
        res_vec.resize(system_data.all_resources.get_number_resources(i));

        auto& mean_vec = mean_time_temp[i][j];
        mean_vec.resize(system_data.all_resources.get_number_resources(i));
      }
    }

    Logger::Debug("Containers resized");

    // loop over all partitions
    for(const auto& [part, part_data] : comp_data.items())
    {
      for(const auto& [res, perf_data] : part_data.items())
      {
        const std::size_t comp_idx = system_data.comp_name_to_idx.at(comp);
        const ResourceType& res_type = system_data.res_name_to_type_and_idx.at(
          res
        ).first;
        const auto res_type_idx = ResIdxFromType(res_type);
        const std::size_t part_idx = system_data.part_name_to_part_idx.at(
          comp + part
        );
        const auto res_idx = system_data.res_name_to_type_and_idx.at(
          res
        ).second;

        std::ostringstream msg;
        msg << "Analyzing component (" << c_idx << ", " << part_idx 
            << ") executed on resource " << res_idx 
            << " of type " << res_type_idx;
        Logger::Debug(msg.str());

        // check that the current partition and resource are compatible
        if(system_data.compatibility_matrix[comp_idx][res_type_idx][part_idx][res_idx])
        {
          // get performance model and average execution time
          const std::string model = perf_data.at("model").get<std::string>();
          const TimeType meanTime = perf_data.at("meanTime").get<TimeType>();

          msg.str("");
          msg << "** Current component has model " << model 
              << " and meanTime " << meanTime;
          Logger::Trace(msg.str());

          // save demand
          if(model == "QTedge" || model == "QTcloud")
          {
            const TimeType d = perf_data.at("demand").get<TimeType>();
            demands_edge_vm_temp[res_type_idx][part_idx][res_idx] = d;
          }

          // save information
          perf_temp[res_type_idx][part_idx][res_idx] = create_PE(
            model, perf_data, system_data, comp_idx, part_idx, res_idx
          );
          Logger::Trace("*** model saved");
          mean_time_temp[res_type_idx][part_idx][res_idx] = meanTime;
          Logger::Trace("*** meanTime saved");
        }
        else // non compatible
        {
          const std::string err_message = "In \
            System::initialize_performance(...) error in allocation of \
            performance for incompatible resource: " + res + "and component " 
            + comp + "with partition " + part + "\n";
          Logger::Error(err_message);
          throw std::logic_error(err_message);
        }
      }
    }

    this->performance.push_back(std::move(perf_temp));
    all_demands.push_back(std::move(demands_edge_vm_temp));
    job_mean_times.push_back(std::move(mean_time_temp));
  }

  QTPE::set_all_demands(std::move(all_demands));
  SystemPE::set_mean_times(std::move(job_mean_times));
}

} // namespace Space4AI
