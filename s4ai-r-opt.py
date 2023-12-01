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

from external import space4ai_logger, space4ai_parser

from typing import Tuple
import argparse
import json
import os


# mount point
MOUNT_POINT = os.getenv("MOUNT_POINT", "/mnt")


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="SPACE4AI-R optimizer")
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--load", 
      help="Input workload", 
      type=float
    )
    parser.add_argument(
      "--on_edge", 
      help="True if the optimizer should consider only edge resources", 
      default=False, 
      action="store_true"
    )
    parser.add_argument(
      "--RG_n_iterations", 
      help="Number of iterations for the RandomGreedy algorithm", 
      type=int, 
      default=1000
    )
    parser.add_argument(
      "--LS_n_iterations", 
      help="Number of iterations for the LocalSearch algorithm", 
      type=int, 
      default=10
    )
    parser.add_argument(
      "--max_num_sols", 
      help="Maximum number of elite solutions to be saved", 
      type=int, 
      default=10
    )
    parser.add_argument(
      "--drop_seed", 
      help="True to avoid using a fixed seed for random number generation",
      default=False, 
      action="store_true"
    )
    parser.add_argument(
      "--verbosity_level", 
      help="Verbosity level for logging", 
      type=str, 
      choices=[
        "INFO", 
        "DEBUG", 
        "TRACE"
      ],
      default="INFO"
    )
    parser.add_argument(
      "--log_on_file", 
      help="True to print logging info to a s4ai-r-LOG.log file",
      default=False, 
      action="store_true"
    )
    args, _ = parser.parse_known_args()
    return args


def convert_verbosity_level(verbosity_level: str, who: str) -> int:
    if who == "python":
        if verbosity_level == "INFO":
          return 0
        elif verbosity_level == "DEBUG":
            return 3
        elif verbosity_level == "TRACE":
            return 7
    elif who == "c++":
        if verbosity_level == "INFO":
            return 2
        elif verbosity_level == "DEBUG":
            return 1
        elif verbosity_level == "TRACE":
            return 0
    else:
        return -1


def load_config_file(config_file: str) -> dict:
    """
    Load configuration file (in json format) into a dictionary
    """
    config = {}
    with open(config_file, "r") as istream:
      config = json.load(istream)
    return config


def write_config_file(config: dict, config_file: str):
    """
    Write configuration dictionary to json file
    """
    with open(config_file, "w") as ostream:
        json.dump(config, ostream, indent=2)


def parse_input_files(
      application_dir: str, 
      logger: space4ai_logger.Logger,
      alternative_deployment: str = None,
      on_edge: bool = False
    ) -> Tuple[str,str]:
    """
    Parse the input YAML files to generate the system description JSON file 
    and the JSON file of the current production deployment
    """
    # define parser
    parser = space4ai_parser.ParserYamlToJson(
      application_dir=application_dir, 
      who="s4ai-r",
      alternative_deployment=alternative_deployment,
      log=space4ai_logger.Logger(
        name="S4AIParser",
        out_stream=logger.out_stream,
        verbose=logger.verbose
      ),
      only_edge=on_edge
    )
    # generate system file
    system_file = parser.make_system_file()
    optimizable = True
    # generate current deployment file
    current_deployment_file = parser.make_current_solution()
    feasible = True
    if on_edge:
        _, feasible, optimizable = parser.filter_current_solution()
    return system_file, current_deployment_file, feasible, optimizable


def write_output_file(
      application_dir: str,
      logger: space4ai_logger.Logger,
      alternative_deployment: str = None
    ):
    """
    Convert the output JSON file with the new production deployment to YAML 
    format
    """
    # generate the new production deployment YAML file
    j2y_parser = space4ai_parser.ParserJsonToYaml(
      application_dir=application_dir, 
      who="s4ai-r",
      alternative_deployment=alternative_deployment,
      log=space4ai_logger.Logger(
        name="S4AIParser",
        out_stream=logger.out_stream,
        verbose=logger.verbose
      )
    )
    j2y_parser.main_function()


def build_optimizer_config(
      system_file: str,
      new_deployment_file: str,
      args: argparse.Namespace
    ) -> str:
    """
    Generate the configuration file for the optimizer and write it to a file
    """
    # generate configuration dictionary
    verbosity_level = convert_verbosity_level(args.verbosity_level, "c++")
    config = {
      "ConfigFiles": [system_file],
      "OutputFiles": [new_deployment_file],
      "Lambda": args.load,
      "Algorithm": {
        "RG_n_iterations": args.RG_n_iterations,
        "LS_n_iterations": args.LS_n_iterations,
        "max_num_sols": args.max_num_sols,
        "reproducibility": not args.drop_seed
      },
      "Logger": {
        "priority": verbosity_level,
        "terminal_stream": True,#not args.log_on_file,
        "file_stream": False,#args.log_on_file
      }
    }
    # write configuration dict to file
    config_file = os.path.join(
      MOUNT_POINT, 
      args.application_dir, 
      "space4ai-r/Config.json"
    )
    write_config_file(config, config_file)
    return config_file


def check_feasibility(solution_file: str) -> bool:
    """
    Check if the new solution is feasible
    """
    # load solution
    feasible = True
    with open(solution_file, "r") as file:
        solution = json.load(file)
        if "feasible" in solution:
            feasible = solution["feasible"]
    return feasible


def check_workload(args: argparse.Namespace, alternative_deployment: str):
    """
    Check if the current workload can be managed by the current deployment
    """
    admissible_load = True
    if alternative_deployment is not None:
        # open the file with the maximum workload
        max_workload_file = os.path.join(
          MOUNT_POINT,
          args.application_dir, 
          "space4ai-r",
          f"Output_max_Lambda_{alternative_deployment}.json"
        )
        with open(max_workload_file, "r") as file:
            max_workload = float(json.load(file)["Lambda"])
            if args.load > max_workload:
                admissible_load = False
    return admissible_load


def main(
      args: argparse.Namespace, 
      logger: space4ai_logger.Logger,
      log_file: str
    ) -> int:
    """
    Main function: generate space4ai-r-optimizer input, run the optimizer, 
    convert the output to a yaml file
    """
    # define space4ai-r optimizer executable
    optimizer = "s4ai-r-optimizer/BUILD/apps/s4air_exe"
    # define the output file
    application_dir = os.path.join(MOUNT_POINT, args.application_dir)
    new_deployment_file = os.path.join(
      application_dir, "space4ai-r/Output.json"
    )
    # get the list of alternative deployments
    alternative_deployments = space4ai_parser.Parser.get_alternative_list(
        application_dir
    )
    # loop over deployments
    for alternative in alternative_deployments:
        logger.log(80*"-")
        # check if the current alternative is not the unique one
        if len(alternative_deployments) > 1:
            alternative_deployment = alternative
        else:
            alternative_deployment = None
        # check maximum admissible workload for the current deployment
        admissible_load = check_workload(args, alternative_deployment)
        if admissible_load:
            # parse input YAML files
            logger.log("Parse input YAML files")
            system_file, _, feasible, optimizable = parse_input_files(
                application_dir=application_dir,
                logger=space4ai_logger.Logger(
                  name="SPACE4AI-R-Optimizer",
                  out_stream=logger.out_stream, 
                  verbose=logger.verbose
                ),
                alternative_deployment=alternative_deployment,
                on_edge=args.on_edge
            )
            if optimizable:
                # generate configuration dictionary for s4ai-r
                logger.log("Generate space4ai-r configuration")
                config_file = build_optimizer_config(
                  system_file=system_file,
                  new_deployment_file=new_deployment_file,
                  args=args
                )
                # run the optimizer
                command = f"{optimizer} {config_file}"
                logger.log(f"Running command `{command}`")
                if args.log_on_file:
                    command += f" &>> {log_file}"
                out = os.system(command)
                # if the run is successful, convert the output file to YAML
                if out == 0:
                    logger.log("Write output YAML file")
                    write_output_file(
                      application_dir=application_dir,
                      logger=space4ai_logger.Logger(
                        name="SPACE4AI-R-Optimizer",
                        out_stream=logger.out_stream, 
                        verbose=logger.verbose
                      ),
                      alternative_deployment=alternative_deployment
                    )
                    # check feasibility and, if feasible, exit the loop
                    feasible = check_feasibility(
                      solution_file=new_deployment_file
                    )
                    if feasible:
                        return out
            else:
                # print unfeasible solution
                unfeasible_solution = {"feasible": False}
                with open(new_deployment_file, "w") as file:
                    json.dump(unfeasible_solution, file, indent=4)
                logger.log("Write output YAML file")
                write_output_file(
                  application_dir=application_dir,
                  logger=space4ai_logger.Logger(
                    name="SPACE4AI-R-Optimizer",
                    out_stream=logger.out_stream, 
                    verbose=logger.verbose
                  ),
                  alternative_deployment=alternative_deployment
                )
                # exit
                out = 0
        else:
            logger.warn("Current load {} is not feasible (deployment: {})".\
                format(args.load, alternative_deployment))
            # print unfeasible solution
            unfeasible_solution = {"feasible": False}
            with open(new_deployment_file, "w") as file:
                json.dump(unfeasible_solution, file, indent=4)
            logger.log("Write output YAML file")
            write_output_file(
              application_dir=application_dir,
              logger=space4ai_logger.Logger(
                name="SPACE4AI-R-Optimizer",
                out_stream=logger.out_stream, 
                verbose=logger.verbose
              ),
              alternative_deployment=alternative_deployment
            )
            # exit
            out = 0
    return out


if __name__ == "__main__":
    # parse input arguments
    args = parse_arguments()
    # initialize logger
    verbosity_level = convert_verbosity_level(args.verbosity_level, "python")
    logger = space4ai_logger.Logger(
      name="SPACE4AI-R-Optimizer",
      verbose=verbosity_level
    )
    log_file = None
    log_stream = None
    if args.log_on_file:
        log_file = os.path.join(
          MOUNT_POINT, 
          args.application_dir, 
          "space4ai-r", 
          "s4ai-r-LOG.log"
        )
        log_stream = open(log_file, "a")
        logger.out_stream = log_stream
    # run and print output
    out = main(args, logger, log_file)
    logger.log(f"S4AI-R optimizer returned output: {out}")
    # close logger stream
    if args.log_on_file:
        log_stream.close()
