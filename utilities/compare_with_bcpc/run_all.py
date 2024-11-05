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
from typing import Tuple
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
  args, _ = parser.parse_known_args()
  return args


def load_thresholds(filename: str) -> dict:
  thresholds = {}
  with open(filename, "r") as istream:
    thresholds = json.load(istream)
  return thresholds


def get_base_config() -> dict:
  base_config = {
    "ConfigFiles": [
      None
    ],
    "OutputFiles": [
      None
    ],
    "Lambda": 1.0,
    "Algorithm": {
      "RG_n_iterations": 1e5,
      "LS_n_iterations": 100,
      "max_num_sols": 10,
      "reproducibility": True
    },
    "Logger": {
      "priority": 2,
      "terminal_stream": True,
      "file_stream": False
    }
  }
  return base_config


def update_config(instance_folder: str, threshold: int) -> Tuple[dict, str]:
  config = get_base_config()
  config["ConfigFiles"][0] = os.path.join(
    instance_folder,
    f"system_description_{threshold}_updated.json"
  )
  config["OutputFiles"][0] = os.path.join(
    instance_folder,
    f"s4air_solution_{threshold}.json"
  )
  log_file = os.path.join(instance_folder, f"s4air_{threshold}.log")
  return config, log_file


def write_config_file(base_folder: str, config: dict):
  config_file = os.path.join(base_folder, "config.json")
  with open(config_file, "w") as ostream:
    ostream.write(json.dumps(config, indent = 2))
  return config_file


def main(base_folder: str):
  # load thresholds
  all_thresholds = load_thresholds(
    os.path.join(base_folder, "thresholds.json")
  )
  # loop over all scenarios
  successful = []
  failed = []
  for scenario, instances in all_thresholds.items():
    print(f"Scenario: {scenario}")
    # loop over all instances
    for instance, thresholds in instances.items():
      print(f"  Instance: {instance}")
      instance_folder = os.path.join(base_folder, scenario, instance)
      # loop over all thresholds
      for threshold in thresholds:
        print(f"    threshold: {threshold}")
        # write configuration file
        config, log_file = update_config(instance_folder, threshold)
        config_file = write_config_file(base_folder, config)
        # run space4ai-r
        exe = "/home/SPACE4AI-R/s4ai-r-optimizer/BUILD/apps/s4air_exe"
        command = f"{exe} {config_file} > {log_file} 2>&1"
        out = os.system(command)
        if out == 0:
          successful.append((scenario, instance, threshold))
        else:
          failed.append((scenario, instance, threshold))
  # write list of successful/failed tests
  with open(os.path.join(base_folder, "successful.txt"), "w") as ostream:
    for exp in successful:
      ostream.write(f"{exp}\n")
  with open(os.path.join(base_folder, "failed.txt"), "w") as ostream:
    for exp in failed:
      ostream.write(f"{exp}\n")


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  main(base_folder)
