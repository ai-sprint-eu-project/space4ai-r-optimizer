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
from datetime import datetime
from typing import Tuple
from parse import parse
from tqdm import tqdm
import pandas as pd
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Run SPACE4AI-R on the instances considered by BCPC"
  )
  parser.add_argument(
    "--application_dir", 
    help="Path to the base folder", 
    type=str
  )
  parser.add_argument(
    "-n", "--n_components", 
    help="Number of components to consider", 
    type=int,
    nargs="+"
  )
  parser.add_argument(
    "-i", "--n_instances", 
    help="Number of instances to consider (for each scenario)",
    default="all"
  )
  parser.add_argument(
    "-c", "--constraints", 
    help="Type of constraint to consider (for each instance)",
    choices=["light", "strict", "mid", "all"],
    default="all",
    nargs="+"
  )
  args, _ = parser.parse_known_args()
  return args


def load_thresholds(filename: str) -> dict:
  thresholds = {}
  with open(filename, "r") as istream:
    thresholds = json.load(istream)
  return thresholds


def get_base_config(filename: str) -> dict:
  base_config = {}
  with open(filename, "r") as istream:
    base_config = json.load(istream)
  return base_config


def sort_and_filter(thresholds: dict, constraints_rules) -> list:
  # sort by the rescaled values
  sorted_thresholds = pd.DataFrame(thresholds).sort_values("rescaled")
  thresholds_subset = list(sorted_thresholds["original"])
  # extract subset of thresholds to consider according to the constraints rule
  if constraints_rules != "all":
    thresholds_subset = []
    for constraints_rule in constraints_rules:
      if constraints_rule == "light":
        subset = sorted_thresholds[
          sorted_thresholds["rescaled"] >= 60
        ]
        thresholds_subset += list(subset["original"])
      elif constraints_rule == "mid":
        subset = sorted_thresholds[
          (sorted_thresholds["rescaled"] >= 10) & 
          (sorted_thresholds["rescaled"] < 60)
        ]
        thresholds_subset += list(subset["original"])
      elif constraints_rule == "strict":
        subset = sorted_thresholds[
          (sorted_thresholds["rescaled"] < 10)
        ]
        thresholds_subset += list(subset["original"])
  # return the list of original thresholds
  return thresholds_subset


def update_config(
    config: dict, instance_folder: str, output_folder: str, threshold: int
  ) -> Tuple[dict, str]:
  config["ConfigFiles"][0] = os.path.join(
    instance_folder,
    f"system_description_{threshold}_updated.json"
  )
  config["OutputFiles"][0] = os.path.join(
    output_folder,
    f"s4air_solution_{threshold}.json"
  )
  log_file = os.path.join(output_folder, f"s4air_{threshold}.log")
  return config, log_file


def write_config_file(base_folder: str, config: dict):
  config_file = os.path.join(base_folder, "config.json")
  with open(config_file, "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
  return config_file


def main(
    base_folder: str, 
    n_components_list: list, 
    n_instances: str,
    constraints_rules = "all"
  ):
  # load thresholds
  all_thresholds = load_thresholds(
    os.path.join(base_folder, "large_scale/thresholds.json")
  )
  # load base configuration info
  base_config = get_base_config(
    os.path.join(base_folder, "space4air_input.json")
  )
  # build output folder
  constr = "allConstraints"
  if constraints_rules != "all":
    constr = "-".join(constraints_rules) + "Constraints"
  cfg = base_config['Algorithm']
  maxit = f"RG{cfg['RG_n_iterations']}_LS{cfg['LS_n_iterations']}"
  nsol = f"{cfg['max_num_sols']}sol"
  base_output_folder = os.path.join(
    base_folder, f"output_{constr}_{maxit}_{nsol}"
  )
  os.makedirs(base_output_folder, exist_ok = True)
  # loop over all scenarios
  successful = []
  failed = []
  exec_times = []
  for scenario, instances in all_thresholds.items():
    if int(parse("{}Components", scenario)[0]) in n_components_list:
      print(f"Scenario: {scenario}")
      # loop over all instances
      n_instances = int(n_instances) if n_instances != "all" else len(instances)
      for instance, thresholds in instances.items():
        if int(parse("Ins{}", instance)[0]) <= n_instances:
          print(f"  Instance: {instance}")
          instance_folder = os.path.join(
            base_folder, "large_scale", scenario, instance
          )
          output_folder = os.path.join(base_output_folder, scenario, instance)
          os.makedirs(output_folder, exist_ok = True)
          # sort thresholds by rescaled values
          sorted_thresholds = sort_and_filter(thresholds, constraints_rules)
          # loop over all thresholds
          for threshold in tqdm(sorted_thresholds):
            # write configuration file
            config, log_file = update_config(
              base_config, instance_folder, output_folder, threshold
            )
            config_file = write_config_file(base_output_folder, config)
            # run space4ai-r (collect execution times for backup)
            exe = "/home/SPACE4AI-R/s4ai-r-optimizer/BUILD/apps/s4air_exe"
            command = f"{exe} {config_file} > {log_file} 2>&1"
            s = datetime.now()
            out = os.system(command)
            e = datetime.now()
            if out == 0:
              successful.append((scenario, instance, threshold))
            else:
              failed.append((scenario, instance, threshold))
            exec_times.append(
              (scenario, instance, threshold, (e - s).total_seconds())
            )
  # write list of successful/failed tests and execution times
  with open(os.path.join(base_output_folder, "successful.txt"), "w") as ostream:
    for exp in successful:
      ostream.write(f"{exp}\n")
  with open(os.path.join(base_output_folder, "failed.txt"), "w") as ostream:
    for exp in failed:
      ostream.write(f"{exp}\n")
  with open(os.path.join(base_output_folder, "times.txt"), "w") as ostream:
    for exp in exec_times:
      ostream.write(f"{exp}\n")


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  n_components_list = args.n_components
  n_instances = args.n_instances
  constraints_rules = args.constraints
  main(base_folder, n_components_list, n_instances, constraints_rules)
