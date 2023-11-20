#include <filesystem>
#include <fstream>
#include <iostream>

#include "src/s4ai.hpp"

namespace sp = Space4AI;
namespace fs = std::filesystem;
namespace nl = nlohmann;

int
main(int argc, char** argv)
{
  if(argc != 2)
  {
    throw std::invalid_argument(
      "Wrong number of arguments provided. Plese provide just the path of \
      the basic json configuration file"
    );
  }

  // read basic configuration file
  const fs::path basic_config_filepath = argv[1];
  std::ifstream basic_config_file(basic_config_filepath);
  nl::json basic_config;

  if(basic_config_file)
  {
    basic_config = nl::json::parse(basic_config_file);
  }
  else
  {
    std::string err_msg = "Can't open " + 
                          basic_config_filepath.string() + 
                          " file. Make sure that the path is correct, and \
                          the format is json";
    throw std::runtime_error(err_msg);
  }

  // initialize workload
  const auto lambda = basic_config.at("Lambda").get<sp::LoadType>();

  // initialize logger
  Logger::SetPriority(static_cast<LogPriority>(basic_config.at(
    "Logger"
  ).at("priority").get<int>()));
  Logger::EnableTerminalOutput(basic_config.at(
    "Logger"
  ).at("terminal_stream").get<bool>());

  // check that the number of provided system files and solutions is 1
  if(
    basic_config.at("ConfigFiles").size() != 1 ||
    basic_config.at("DTSolutions").size() != 1 ||
    basic_config.at("BinarySearchOutputs").size() != 1
  )
  {
    throw std::length_error(
      "Error in configuration input file: *ConfigFiles* and/or \
      *DTSolutions* and/or *BinarySearchOutputs* dimension is not 1"
    );
  }

  // initialize system
  sp::System system;
  const std::string system_config_file = basic_config.at(
    "ConfigFiles"
  )[0].get<std::string>();
  system.read_configuration_file(
    system_config_file, 
    lambda, 
    1.0
  );

  // load current production deployment
  sp::Solution current_sol(system);
  const std::string solution_config_file = basic_config.at(
    "DTSolutions"
  )[0].get<std::string>();
  current_sol.read_solution_from_file(solution_config_file, system);

  // QoS constraints check
  bool feasible = current_sol.check_QoS_constraints(system);

  // print the output to a file
  if(feasible)
  {
    current_sol.print_solution(
      system, basic_config.at("BinarySearchOutputs")[0].get<std::string>()
    );
  }
  else
  {
    sp::Solution::print_unfeasible_solution(
      system, basic_config.at("BinarySearchOutputs")[0].get<std::string>()
    );
  }

  Logger::Info("check feasibility completed");

  return 0;
}