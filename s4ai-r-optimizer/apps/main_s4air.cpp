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

  // initialize algorithm parameters
  const auto& algo_config = basic_config.at("Algorithm");
  const size_t rg_n_iterations=algo_config.at("RG_n_iterations").get<size_t>();
  const size_t ls_n_iterations=algo_config.at("LS_n_iterations").get<size_t>();
  const size_t max_num_sols = algo_config.at("max_num_sols").get<size_t>();
  const bool reproducibility = algo_config.at("reproducibility").get<bool>();
  const auto lambda = basic_config.at("Lambda").get<sp::LoadType>();

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
      "Error in configuration input file: *ConfigFiles* dimension is not 1"
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
    energy_cost_pct
  );

  // initialize RandomGreedy algorithm (used to create a baseline 
  // for LocalSearch)
  sp::RandomGreedy rg;
  sp::SelectedResources curr_sol_res;
  bool initial_solution_exists = false;

  // load current production deployment (if available)
  if (basic_config.contains("DTSolutions"))
  {
    // check that the number of provided solution files is 1
    if (basic_config.at("DTSolutions").size() != 1)
    {
      throw std::length_error(
        "Error in configuration input file: *DTSolutions* dimension is not 1"
      );
    }

    sp::Solution current_sol(system);
    const std::string solution_config_file = basic_config.at(
      "DTSolutions"
    )[0].get<std::string>();
    current_sol.read_solution_from_file(solution_config_file, system);

    // extract the currently selected resources
    curr_sol_res = current_sol.get_selected_resources();
    initial_solution_exists = true;

    // fix the selected resources in RandomGreedy
    rg.set_selected_resources(curr_sol_res);
  }
  
  // run RandomGreedy algorithm to create a baseline for LocalSearch
  const auto& new_rg_elite_result = rg.random_greedy(
    system, 
    rg_n_iterations, 
    max_num_sols, 
    reproducibility
  );

  // check the solution feasibility --> TODO
  bool feasible = true;

  // run LocalSearch algorithm
  sp::LocalSearchManager ls_manager(
    new_rg_elite_result, 
    system, 
    reproducibility, 
    ls_n_iterations, 
    max_num_sols
  );

  if (initial_solution_exists)
    ls_manager.set_selected_resources(curr_sol_res);

  ls_manager.run();
  const auto& ls_elite_result = ls_manager.get_ls_elite_result();

  // check the solution feasibility --> TODO
  feasible = true;

  // print the final solution to a file
  if(feasible)
  {
    if(basic_config.at("OutputFiles").size() != 1)
    {
      throw std::length_error(
        "Error in configuration input file: *OutputFiles* dimension must be 1"
      );
    }
    ls_elite_result.print_solution(
      system, basic_config.at("OutputFiles")[0].get<std::string>()
    );
  }

  Logger::Info("s4ai-r execution completed");

  return 0;
}