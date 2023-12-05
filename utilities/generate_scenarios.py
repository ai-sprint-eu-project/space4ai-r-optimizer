"""
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
"""

from external import space4ai_logger

from generate_scenario import main as generate_scenario

from typing import Tuple
import argparse
import json
import copy
import sys
import os


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
      description="Testing Scenarios Generation"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--lambda_max", 
      help="Maximum workload values", 
      type=float,
      nargs="+"
    )
    parser.add_argument(
      "--seed", 
      help="Seed for random number generation", 
      type=int,
      default=4850
    )
    parser.add_argument(
      "-v", "--verbose", 
      help="Verbosity level", 
      type=int,
      default=0
    )
    args, _ = parser.parse_known_args()
    return args


def read_configuration_file(config_file: str) -> dict:
    config = {}
    with open(config_file, "r") as istream:
        config = json.load(istream)
    return config


def check_parameters_grid(parameters_grid: dict) -> Tuple[bool, str, int]:
    length = 1
    for parameter, values in parameters_grid.items():
        if length == 1:
            length = len(values)
        else:
            if len(values) != length and len(values) != 1:
                return (False, parameter, length)
    return (True, "", length)


def extract_parameters(parameters_grid: dict, n: int):
    parameters = {}
    # loop over the parameters grid
    for parameter, values in parameters_grid.items():
        # get the current value
        value = values[0] if len(values) == 1 else values[n]
        # add the parameter to the dictionary
        if "-" in parameter:
            outer, inner = parameter.split("-")
            if outer not in parameters:
                parameters[outer] = {}
            parameters[outer][inner] = value
        else:
            parameters[parameter] = value
    return parameters


def merge_configuration_dictionaries(
      base_config: dict, 
      parameters: dict, 
      workload: float,
      scenario_dir: str
    ):
    # merge base configuration dictionary with current parameters
    config = copy.deepcopy(base_config)
    for key, value in parameters.items():
        if key not in config:
            config[key] = value
        else:
            for inner_key, inner_value in value.items():
                config[key][inner_key] = inner_value
    # add workload
    config["lambda"] = workload
    # write to file
    with open(os.path.join(scenario_dir, "config.json"), "w") as ostream:
        json.dump(config, ostream, indent=2)


def main(
      application_dir: str, 
      lambda_max: list,
      seed: int, 
      logger: space4ai_logger.Logger
    ):
    # read base configuration file
    config_file = os.path.join(application_dir, "base_config.json")
    config = read_configuration_file(config_file)
    base_config = config["base_config"]
    parameters_grid = config["parameters_grid"]
    # check if the parameters grid is valid
    valid, culprit, n_scenarios = check_parameters_grid(parameters_grid)
    if not valid:
        logger.err(f"Length of parameter {culprit} is invalid")
        sys.exit(1)
    # loop over all workload values
    for workload in lambda_max:
        workload_dir = os.path.join(application_dir, f"Lambda_{workload}")
        os.makedirs(workload_dir, exist_ok=True)
        logger.log(f"Generating scenarios for lambda_max = {workload}")
        # loop over all scenarios
        for n in range(n_scenarios):
            scenario_dir = os.path.join(workload_dir, f"Scenario{n}")
            os.makedirs(scenario_dir, exist_ok=True)
            logger.log(
              f"Generating scenario {n}. Saving files in {scenario_dir}"
            )
            # extract dictionary of parameters
            parameters = extract_parameters(parameters_grid, n)
            # write configuration file
            merge_configuration_dictionaries(
              base_config, parameters, workload, scenario_dir
            )
            # generate scenario
            generate_scenario(scenario_dir, seed, logger)
            seed += 1000


if __name__ == "__main__":
    # parse arguments
    args = parse_arguments()
    application_dir = args.application_dir
    lambda_max = args.lambda_max
    seed = args.seed
    verbose = args.verbose
    # run
    logger = space4ai_logger.Logger(name="GenerateScenarios", verbose=verbose)
    main(application_dir, lambda_max, seed, logger)
