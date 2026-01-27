from call_checkfeasibility_api import main as call_api
from compare_heuristics import (
    build_uheur_directory,
    convert_verbosity_level,
    get_data_list
)
from external import space4ai_logger

from shutil import copy as copyfile
from datetime import datetime
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
      description="Check solutions feasibility along a bandwidth trace"
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
      "--epsilon", 
      help="Binary search tolerance", 
      type=float,
      default=None
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


def check_feasibility(
        application_dir: str,
        solution_file: str, 
        workload: float, 
        bandwidth: float,
        bdw_min: float,
        epsilon: float,
        method: str
    ):
    # create i/o directories
    dirname = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S.%f')
    input_dir = os.path.join(MOUNT_POINT, "input", dirname)
    os.makedirs(input_dir, exist_ok = True)
    output_dir = os.path.join(MOUNT_POINT, "output", dirname, method)
    os.makedirs(output_dir, exist_ok = True)
    # copy system file
    copyfile(
        os.path.join(application_dir, "SystemFile.json"),
        os.path.join(input_dir, "SystemFile.json")
    )
    # copy (and rename) solution file
    solname = os.path.basename(solution_file).replace("Solution-", "")
    copyfile(solution_file, os.path.join(output_dir, solname))
    # call check-feasibility api
    feasible = call_api(
        check_home = False,
        application_dir = dirname,
        verbosity_level = verbosity_level,
        min_load = workload,
        max_load = workload,
        min_bandwidth = bdw_min,
        max_bandwidth = bandwidth,
        epsilon = epsilon,
        method = method,
        aisprint = False
    )
    return feasible, dirname


def main(
      args: argparse.Namespace, 
      logger: space4ai_logger.Logger
    ) -> int:
    # identify results directories
    s4air_dir = os.path.join(args.application_dir, "s4air")
    uheur_dir = build_uheur_directory(args)
    # get list of workload and bandwidth values
    lambdas = get_data_list(args.application_dir, "Lambda")
    bandwidths = get_data_list(args.application_dir, "Bandwidth")
    bdw_ext = get_data_list(
        os.path.join(args.application_dir, "../.."), 
        "Bandwidth", 
        "Extended"
    )
    bdw_step = len(bdw_ext[1:]) // len(bandwidths[1:])
    # loop over workloads/bandwidths (skip design-time)
    feasible = []
    for i, (workload, bandwidth) in enumerate(zip(lambdas[1:],bandwidths[1:])):
        logger.log(
            f"Considering workload {workload}; bandwidth {bandwidth}"
        )
        s4air_solution_file = os.path.join(
            s4air_dir, f"Solution-lambda_{workload}-bandwidth_{bandwidth}.json"
        )
        uheur_solution_file = os.path.join(
            uheur_dir, f"Solution-lambda_{workload}-bandwidth_{bandwidth}.json"
        )
        # identify minimum bandwidth value in the extended trace
        bdw_min = min(bdw_ext[i:i+bdw_step])
        # check feasibility
        s_feasible, s_dirname = check_feasibility(
            application_dir = args.application_dir,
            solution_file = s4air_solution_file,
            workload = workload,
            bandwidth = bandwidth,
            bdw_min = bdw_min,
            epsilon = args.epsilon,
            method = "space4air"
        )
        u_feasible, u_dirname = check_feasibility(
            application_dir = args.application_dir,
            solution_file = uheur_solution_file,
            workload = workload,
            bandwidth = bandwidth,
            bdw_min = bdw_min,
            epsilon = args.epsilon,
            method = "uheur"
        )
        feasible.append({
            "bandwidth": bandwidth,
            "min_bandwidth_interval": bdw_min,
            "s4air_feasible": s_feasible,
            "s4air_dirname": s_dirname,
            "uheur_feasible": u_feasible,
            "uheur_dirname": u_dirname
        })
    with open(
          os.path.join(args.application_dir, "trace_feasibility.txt"), "w"
        ) as ostream:
        for line in feasible:
            ostream.write(f"{line}\n")



if __name__ == "__main__":
    # parse input arguments
    args = parse_arguments()
    # initialize logger
    verbosity_level = convert_verbosity_level(args.verbosity_level, "python")
    logger = space4ai_logger.Logger(
      name="SPACE4AI-R-CheckTraceFeasibility",
      verbose=verbosity_level
    )
    # run and print output
    out = main(args, logger)
    logger.log(f"CheckTraceFeasibility returned output: {out}")
