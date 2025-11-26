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

  // initialize workload and bandwidth
  const auto lambda = basic_config.at("Lambda").get<sp::LoadType>();
  const auto bandwidth = basic_config.at("Bandwidth").get<double>();

  // initialize algorithm parameters
  const auto& algo_config = basic_config.at("Algorithm");
  if (!algo_config.contains("UtilizationHeuristic"))
  {
    throw std::invalid_argument(
      "Missing UtilizationHeuristic parameters in configuration file"
    );
  }
  const auto& uh_config = algo_config.at("UtilizationHeuristic");
  const std::string rule = uh_config.at("rule").get<std::string>();
  const double min_utilization = uh_config.at("min_utilization").get<double>();
  const double max_utilization = uh_config.at("max_utilization").get<double>();

  const double energy_cost_pct = 1.0; // WHAT THE HELL IS THIS?!?

  // initialize logger
  Logger::SetPriority(static_cast<LogPriority>(basic_config.at(
    "Logger"
  ).at("priority").get<int>()));
  Logger::EnableTerminalOutput(basic_config.at(
    "Logger"
  ).at("terminal_stream").get<bool>());

  // check that the number of provided system files is 1
  if(basic_config.at("ConfigFiles").size() != 1)
  {
    throw std::length_error(
      "Error in configuration file: *ConfigFiles* dimension is not 1"
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
    bandwidth,
    energy_cost_pct
  );

  // initialize UtilizationHeuristic algorithm
  sp::UtilizationHeuristic uh(system, rule);

  if (rule != "fixed")
  {
    if (rule == "percentage")
    {
      if (uh_config.contains("decr_percentage"))
        uh.set_decr_percentage(uh_config.at("decr_percentage").get<double>());
      if (uh_config.contains("incr_percentage"))
        uh.set_incr_percentage(uh_config.at("incr_percentage").get<double>());
    }
    else
      throw std::invalid_argument(
        "Unknown rule " + rule
      );
  }

  // load design-time resources (if available)
  sp::Solution dt_sol(system);
  if (basic_config.contains("DTSolutions"))
  {
    // check that the number of provided solution files is 1
    if (basic_config.at("DTSolutions").size() != 1)
    {
      throw std::length_error(
        "Error in configuration file: *DTSolutions* dimension is not 1"
      );
    }

    const std::string solution_config_file = basic_config.at(
      "DTSolutions"
    )[0].get<std::string>();
    dt_sol.read_solution_from_file(solution_config_file, system);

    // fix the selected resources in UtilizationHeuristic
    uh.set_selected_resources(dt_sol.get_selected_resources());
  }

  // load current production deployment (if available)
  sp::Solution current_sol(system);
  if (basic_config.contains("CurrentSolutions"))
  {
    // check that the number of provided solution files is 1
    if (basic_config.at("CurrentSolutions").size() != 1)
    {
      throw std::length_error(
        "Error in configuration file: *CurrentSolutions* dimension is not 1"
      );
    }
    Logger::Info("Using the current solution as starting point");

    // read current solution
    const std::string solution_config_file = basic_config.at(
      "CurrentSolutions"
    )[0].get<std::string>();
    current_sol.read_solution_from_file(solution_config_file, system);
  }
  else
  {
    Logger::Info("Using the design-time solution as starting point");
    current_sol = dt_sol;
  }
  
  // run UtilizationHeuristic algorithm
  const auto& uh_elite_result = uh.run(
    current_sol, 
    min_utilization, 
    max_utilization
  );

  // print the final solution to a file
  if(basic_config.at("OutputFiles").size() != 1)
  {
    throw std::length_error(
      "Error in configuration file: *OutputFiles* dimension must be 1"
    );
  }
  uh_elite_result.print_solution(
    system, basic_config.at("OutputFiles")[0].get<std::string>()
  );

  Logger::Info("UtilizationHeuristic execution completed");

  return 0;
}