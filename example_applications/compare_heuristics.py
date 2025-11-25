from external import space4ai_logger

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
    parser = argparse.ArgumentParser(
      description="SPACE4AI-R vs UtilizationHeuristic"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--heuristic_rule", 
      help="Rule to be followed by Utilization Heuristic ", 
      type=str,
      choices=[
          "fixed",
          "percentage"
      ],
      default="fixed"
    )
    parser.add_argument(
      "--min_utilization", 
      help="Minimum utilization threshold", 
      type=float,
      default=0.4
    )
    parser.add_argument(
      "--max_utilization", 
      help="Maximum utilization threshold", 
      type=float,
      default=0.5
    )
    parser.add_argument(
      "--decr_percentage", 
      help="Number of instances decrease percentage", 
      type=float,
      default=0.1
    )
    parser.add_argument(
      "--incr_percentage", 
      help="Number of instances increase percentage", 
      type=float,
      default=0.6
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


def get_workload_list(application_dir: str) -> list:
    """
    Get list of workload values from file
    """
    lambdas = []
    workload_file = os.path.join(application_dir, "LambdaValues.json")
    with open(workload_file, "r") as istream:
        lambdas = json.load(istream)["LambdaVec"]
    return lambdas


def write_config_file(config: dict, config_file: str):
    """
    Write configuration dictionary to json file
    """
    with open(config_file, "w") as ostream:
        json.dump(config, ostream, indent=2)


def build_optimizer_config(
      args: argparse.Namespace,
      workload: float,
      dt_solution_file: str
    ) -> dict:
    """
    Generate the base configuration dictionary for the optimizer
    """
    # generate configuration dictionary
    verbosity_level = convert_verbosity_level(args.verbosity_level, "c++")
    config = {
      "ConfigFiles": [os.path.join(args.application_dir, "SystemFile.json")],
      "DTSolutions": [dt_solution_file],
      "Lambda": workload,
      "Logger": {
        "priority": verbosity_level,
        "terminal_stream": True,
        "file_stream": False,
      }
    }
    return config


def build_s4air_config(
      config: dict, 
      workload: float,
      dir: str
    ) -> str:
    """
    Complete the configuration file for space4ai-r and write it to a file
    """
    config["OutputFiles"] = [os.path.join(dir, f"Lambda_{workload}.json")]
    config["Algorithm"] = {
      "RG_n_iterations": 100,
      "LS_n_iterations": 10,
      "max_num_sols": 10,
      "reproducibility": True
    }
    # write configuration dict to file
    config_file = os.path.join(MOUNT_POINT, dir, "Config.json")
    write_config_file(config, config_file)
    return config_file


def build_uheur_config(
      config: dict, 
      args: argparse.Namespace,
      workload: float,
      dir: str,
      current_solution_file: str = None
    ) -> str:
    """
    Complete the configuration file for utilization heuristic and write it 
    to a file
    """
    config["OutputFiles"] = [os.path.join(dir, f"Lambda_{workload}.json")]
    config["Algorithm"] = {
      "UtilizationHeuristic": {
        "rule": args.heuristic_rule,
        "min_utilization": args.min_utilization,
        "max_utilization": args.max_utilization
      }
    }
    # add percentages to increase/decrease the number of resources if the 
    # rule is not 'fixed'
    if args.heuristic_rule == "percentage":
        config["Algorithm"]["UtilizationHeuristic"][
          "decr_percentage"
        ] = args.decr_percentage
        config["Algorithm"]["UtilizationHeuristic"][
          "incr_percentage"
        ] = args.incr_percentage
    # add current solution file (if available)
    if current_solution_file is not None:
        config["CurrentSolutions"] = [os.path.join(dir, current_solution_file)]
    # write configuration dict to file
    config_file = os.path.join(MOUNT_POINT, dir, "Config.json")
    write_config_file(config, config_file)
    return config_file


def get_current_solution(
      application_dir: str,
      lambdas: list, 
      i: int, 
      last_feasible: str, 
      logger: space4ai_logger.Logger
    ) -> Tuple[str, str]:
    current_solution_file = None
    if i > 0:
        current_solution_file = f"Lambda_{lambdas[i]}.json"
        logger.log(f"Current solution is: {current_solution_file}")
        # check if file exists
        filepath = os.path.join(application_dir, current_solution_file)
        if os.path.exists(filepath):
            logger.log("Solution exists")
            # check solution feasibility
            feasible = False
            with open(filepath, "r") as istream:
                feasible = json.load(istream)["feasible"]
            # if the solution is not feasible, consider the last feasible
            if not feasible:
                # current_solution_file = last_feasible
                # logger.log(
                #   f"Solution is not feasible. Considering {last_feasible}"
                # )
                logger.log("Solution is not feasible")
            else:
                last_feasible = current_solution_file
                logger.log("Solution is feasible")
    return current_solution_file, last_feasible


def main(
      args: argparse.Namespace, 
      logger: space4ai_logger.Logger
    ) -> int:
    # define space4ai-r optimizer executable
    s4air_optimizer = "s4ai-r-optimizer/BUILD/apps/s4air_exe"
    # define utilization heuristic executable
    uheur_optimizer = "s4ai-r-optimizer/BUILD/apps/uheur_exe"
    # get list of workload values
    lambdas = get_workload_list(args.application_dir)
    dtw = lambdas[0]
    dt_solution_file = os.path.join(args.application_dir, f"Lambda_{dtw}.json")
    # loop over workloads (skip design-time)
    global_output = 0
    last_feasible = None
    for i, workload in enumerate(lambdas[1:]):
        logger.log(f"Optimizing workload {workload}")
        # define base configuration dictionary
        config = build_optimizer_config(args, workload, dt_solution_file)
        # define directory and configuration file for space4ai-r
        s4air_dir = os.path.join(args.application_dir, "s4air")
        os.makedirs(s4air_dir, exist_ok=True)
        s4air_config_file = build_s4air_config(config, workload, s4air_dir)
        logger.log(f"Written configuration file: {s4air_config_file}", 4)
        # define directory and configuration file for utilization heuristic
        uheur_dir = os.path.join(args.application_dir, "uheur")
        os.makedirs(uheur_dir, exist_ok=True)
        current_solution_file, last_feasible = get_current_solution(
          uheur_dir, lambdas, i, last_feasible, logger
        )
        uheur_config_file = build_uheur_config(
          config, 
          args,
          workload, 
          uheur_dir,
          current_solution_file
        )
        logger.log(f"Written configuration file: {uheur_config_file}", 4)
        # run space4ai-r
        command = f"{s4air_optimizer} {s4air_config_file}"
        logger.log(f"Running command `{command}`")
        out = os.system(command)
        logger.log(f"SPACE4AI-R optimizer returned output {out}")
        global_output += out
        # run utilization heuristic
        command = f"{uheur_optimizer} {uheur_config_file}"
        logger.log(f"Running command `{command}`")
        out = os.system(command)
        logger.log(f"UtilizationHeuristic returned output {out}")
        global_output += out
    return global_output


if __name__ == "__main__":
    # parse input arguments
    args = parse_arguments()
    # initialize logger
    verbosity_level = convert_verbosity_level(args.verbosity_level, "python")
    logger = space4ai_logger.Logger(
      name="SPACE4AI-R-CompareHeuristics",
      verbose=verbosity_level
    )
    # run and print output
    out = main(args, logger)
    logger.log(f"CompareHeuristics returned output: {out}")
