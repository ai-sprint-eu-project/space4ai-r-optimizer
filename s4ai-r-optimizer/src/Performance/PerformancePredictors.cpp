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
* \file PerformancePredictors.cpp
*
* \brief definition of the methods of the class Pacsltk
*
* \author Randeep Singh
*/

#include <cpr/cpr.h>
#include <cstdlib>
#include <algorithm>

#include "external/chrono/chrono.hpp"
#include "src/Performance/PerformancePredictors.hpp"
#include "src/Logger.hpp"


namespace Space4AI
{

/**
 * Pacsltk
*/

TimeType
Pacsltk::predict(const nlohmann::json& features) const
{
  std::string base_url = std::getenv("PACSLTK_URL");
  std::string port = std::getenv("PACSLTK_PORT");
  std::string url = "http://" + base_url + ":" + port + "/pacsltk";

  cpr::Response r = cpr::Post(
    cpr::Url{url},
    cpr::Body{nlohmann::to_string(features)},
    cpr::Header{{"Content-Type", "text/plain"}});

  if(r.status_code != 200)
  {
    std::ostringstream err_msg;
    err_msg << "Connection to " << url << " returned code " << r.status_code;
    Logger::Error(err_msg.str());
    throw std::runtime_error(err_msg.str());
  }

  return std::stod(r.text);

}

/**
 * aMLLibrary
*/

TimeType
aMLLibrary::predict(const nlohmann::json& features) const
{
  std::string base_url = std::getenv("AMLLIBRARY_URL");
  std::string port = std::getenv("AMLLIBRARY_PORT");
  std::string url = "http://" + base_url + ":" + port + "/amllibrary/predict";

  cpr::Response r = cpr::Post(
    cpr::Url{url},
    cpr::Body{features.dump()},
    cpr::Header{{"Content-Type", "application/json"}}
  );

  if(r.status_code != 201)
  {
    std::ostringstream err_msg;
    err_msg << "Connection to " << url << " returned code " << r.status_code;
    Logger::Error(err_msg.str());
    throw std::runtime_error(err_msg.str());
  }

  std::string rt = r.text;
  rt.erase(remove(rt.begin(), rt.end(), '['), rt.end());
  rt.erase(remove(rt.begin(), rt.end(), ']'), rt.end());
  rt.erase(remove(rt.begin(), rt.end(), '"'), rt.end());
  return std::stod(rt);

}

} // namespace Space4AI
