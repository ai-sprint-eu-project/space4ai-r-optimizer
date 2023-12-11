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
* \file PerformanceModels.cpp
*
* \brief Defines methods of the classes defined in PerformanceModels.hpp
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#include "src/Performance/PerformanceModels.hpp"

namespace Space4AI
{

/**
 * QTPE
*/

TimeType
QTPE::predict(
  size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
  const SystemData& system_data, const SolutionData& solution_data
) const
{
  TimeType response_time{0.0};
  const double utilization = compute_utilization(
    res_type, res_idx, system_data, solution_data
  );

  if(utilization > 1) // This is en error! utilization cannot be bigger than -1;
  {
    std::ostringstream err_msg;
    err_msg << "QTPE::predict(): Utilization > 1 of res " << res_idx
            << " with type idx " << ResIdxFromType(res_type) << std::endl;
    Logger::Debug(err_msg.str());
    response_time = -1.;  // manage error in the caller function
  }
  else
  {
    TimeType demand = all_demands[comp_idx][ResIdxFromType(res_type)][part_idx][res_idx];
    response_time = demand / (1 - utilization);
  }

  return response_time;
}

double
QTPE::compute_utilization(
  ResourceType res_type, size_t res_idx,
  const SystemData& system_data, const SolutionData& solution_data
) const
{
  double utilization{0.0};
  const size_t type_idx = ResIdxFromType(res_type);
  const auto& components = system_data.get_components();
  const auto& used_resources = solution_data.get_used_resources();
  const auto& y_hat = solution_data.get_y_hat();

  for(std::size_t c = 0; c < components.size(); ++c)
  {
    for(const auto& [p_idx, r_type_idx, r_idx] : used_resources[c])
    {
      if ((type_idx == r_type_idx) && (r_idx == res_idx))
      {
        TimeType d = all_demands[c][type_idx][p_idx][r_idx];
        LoadType l = components[c].get_partition(p_idx).get_part_lambda();
        size_t n = y_hat[c][type_idx][p_idx][r_idx];
        utilization += (d * l / n);
      }
    }
  }

  return utilization;
}

/**
 * FaasPacsltkPE
*/

TimeType
FaasPacsltkPE::predict(
  size_t comp_idx, size_t part_idx, ResourceType, size_t res_idx,
  const SystemData& system_data, const SolutionData&
) const
{
  const auto& components = system_data.get_components();
  const auto& all_resources = system_data.get_all_resources();
  const auto& partition = components[comp_idx].get_partition(part_idx);
  const auto& resource = all_resources.get_resource<ResourceType::Faas>(res_idx);

  // initialize features json object
  nlohmann::json features;
  features["arrival_rate"] = partition.get_part_lambda();
  features["idle_time_before_kill"] = resource.get_idle_time_before_kill();
  features["warm_service_time"] = this->demandWarm;
  features["cold_service_time"] = this->demandCold;

  // predict
  return this->predictor.predict(features);
}

/**
 * FaasPacsltkStaticPE
*/

FaasPacsltkStaticPE::FaasPacsltkStaticPE(
  const std::string& keyword_,
  bool allows_colocation_,
  TimeType demandWarm_,
  TimeType demandCold_,
  TimeType idle_time_before_kill,
  LoadType part_lambda
):
  FaasPE(keyword_, allows_colocation_, demandWarm_, demandCold_),
  demand(-1)
{
  // initialize features json object
  nlohmann::json features;
  features["arrival_rate"] = part_lambda;
  features["idle_time_before_kill"] = idle_time_before_kill;
  features["warm_service_time"] = this->demandWarm;
  features["cold_service_time"] = this->demandCold;
  
  // predict
  demand = Pacsltk::Instance().predict(features);
}

TimeType
FaasPacsltkStaticPE::predict(
  size_t, size_t, ResourceType, size_t,
  const SystemData&, const SolutionData&
) const
{
  return this->demand;
}

/**
 * FaasaMLLibraryPE
*/

TimeType
FaasaMLLibraryPE::predict(
  size_t, size_t, ResourceType, size_t,
  const SystemData&, const SolutionData&
) const
{

  // initialize features json object
  nlohmann::json features;
  features["regressor"] = this->regressor_file;
  features["df"] = nlohmann::json();

  // predict
  return this->predictor.predict(features);
}

/**
 * CoreBasedaMLLibraryPE
*/

TimeType
CoreBasedaMLLibraryPE::predict(
  size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
  const SystemData& system_data, const SolutionData& solution_data
) const
{
  // get number of instances
  const size_t type_idx = ResIdxFromType(res_type);
  const auto& y_hat = solution_data.get_y_hat();
  const size_t n = y_hat[comp_idx][type_idx][part_idx][res_idx];

  // get number of cores per instance
  const auto& all_resources = system_data.get_all_resources();
  size_t cores_per_res = 0;
  if(type_idx == ResIdxFromType(ResourceType::Edge))
  {
    const auto& res = all_resources.get_resource<ResourceType::Edge>(res_idx);
    cores_per_res = res.get_n_cores();  
  }
  else if(type_idx == ResIdxFromType(ResourceType::VM))
  {
    const auto& res = all_resources.get_resource<ResourceType::VM>(res_idx);
    cores_per_res = res.get_n_cores();  
  }
  else
  {
    Logger::Debug("CoreBasedaMLLibraryPE unavailable for Faas");
    throw std::runtime_error("CoreBasedaMLLibraryPE unavailable for Faas");
  }

  // compute total number of cores
  const size_t cores = n * cores_per_res;

  // initialize features json object
  nlohmann::json features;
  features["regressor"] = this->regressor_file;
  features["df"] = nlohmann::json();
  features["df"]["cores"] = std::vector<size_t>({cores});

  // predict
  return this->predictor.predict(features);
}

/**
 * LambdaBasedaMLLibraryPE
*/

TimeType
LambdaBasedaMLLibraryPE::predict(
  size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
  const SystemData& system_data, const SolutionData& solution_data
) const
{
  // get number of instances
  const size_t type_idx = ResIdxFromType(res_type);
  const auto& y_hat = solution_data.get_y_hat();
  const size_t n = y_hat[comp_idx][type_idx][part_idx][res_idx];

  // get number of cores per instance
  const auto& all_resources = system_data.get_all_resources();
  size_t cores_per_res = 0;
  if(type_idx == ResIdxFromType(ResourceType::Edge))
  {
    const auto& res = all_resources.get_resource<ResourceType::Edge>(res_idx);
    cores_per_res = res.get_n_cores();  
  }
  else if(type_idx == ResIdxFromType(ResourceType::VM))
  {
    const auto& res = all_resources.get_resource<ResourceType::VM>(res_idx);
    cores_per_res = res.get_n_cores();  
  }
  else
  {
    Logger::Debug("LambdaBasedaMLLibraryPE unavailable for Faas");
    throw std::runtime_error("LambdaBasedaMLLibraryPE unavailable for Faas");
  }

  // compute total number of cores
  const size_t cores = n * cores_per_res;

  // get component throughput
  LoadType part_lambda = system_data.get_component(comp_idx).get_partition(
    part_idx
  ).get_part_lambda();

  // initialize features json object
  nlohmann::json features;
  features["regressor"] = this->regressor_file;
  features["df"] = nlohmann::json();
  features["df"]["cores"] = std::vector<size_t>({cores});
  features["df"]["observed_throughput"] = part_lambda;

  // predict
  return this->predictor.predict(features);
}

} //namespace Space4AI
