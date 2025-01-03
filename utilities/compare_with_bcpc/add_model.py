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
from parse import parse
from numpy import inf
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Add model to BCPC input files"
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
  args, _ = parser.parse_known_args()
  return args


def rescale(
    val: float, 
    in_min: float, 
    in_max: float, 
    out_min: float, 
    out_max: float
  ) -> float:
  """
  Rescale value from the original interval [in_min, in_max] to the new 
  range [out_min, out_max]
  """
  return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def update_system_file(filename: str) -> str:
  system = {}
  with open(filename, "r") as istream:
    system = json.load(istream)
  #
  performance = {}
  for c, c_data in system["Performance"].items():
    if c not in performance:
      performance[c] = {}
      for h, h_data in c_data.items():
        if h not in performance[c]:
          performance[c][h] = {}
          for f, f_data in h_data.items():
            performance[c][h][f] = {"model": "FAASFIXED", **f_data}
  system["Performance"] = performance
  #
  dirname, basename = os.path.split(filename)
  new_filename = os.path.join(dirname, f"{basename.split('.')[0]}_updated.json")
  with open(new_filename, "w") as ostream:
    ostream.write(json.dumps(system, indent = 2))
  return new_filename


def main(base_folder: str, n_components_list: list):
  # loop over scenarios
  thresholds = {}
  for n_components in n_components_list:
    scenario = f"{n_components}Components"
    print(f"Processing scenario {scenario}")
    scenario_folder = os.path.join(base_folder, scenario)
    thresholds[scenario] = {}
    # loop over instances
    for foldername in os.listdir(scenario_folder):
      instance_id = parse("Ins{}", foldername)
      if instance_id is not None:
        instance_id = int(instance_id[0])
        print(f"  processing instance {instance_id}")
        instance_folder = os.path.join(scenario_folder, foldername)
        # loop over instance files
        thresholds[scenario][f"Ins{instance_id}"] = {
          "original": [],
          "rescaled": []
        }
        tmin = inf
        tmax = 0.0
        for filename in os.listdir(instance_folder):
          if filename.startswith("system_description"):
            tokens = filename.split("_")
            if len(tokens) == 3 and "updated" not in tokens[-1]:
              t = int(tokens[-1].split(".")[0])
              thresholds[scenario][f"Ins{instance_id}"]["original"].append(t)
              tmin = t if t < tmin else tmin
              tmax = t if t > tmax else tmax
              _ = update_system_file(os.path.join(instance_folder, filename))
        # rescale thresholds
        original = thresholds[scenario][f"Ins{instance_id}"]["original"]
        thresholds[scenario][f"Ins{instance_id}"]["rescaled"] = [
          rescale(t, tmin, tmax, 0, 100) for t in original
        ]
  # write all thresholds (load existing first, if any)
  thresholds_filename = os.path.join(base_folder, "thresholds.json")
  if os.path.exists(thresholds_filename):
    with open(thresholds_filename, "r") as istream:
      existing_thresholds = json.load(istream)
      # merge
      for key, val in existing_thresholds.items():
        if key not in thresholds:
          thresholds[key] = val
        else:
          for ikey, ival in val.items():
            if ikey not in thresholds[key]:
              thresholds[key][ikey] = ival
  with open(thresholds_filename, "w") as ostream:
    ostream.write(json.dumps(thresholds, indent = 2))


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  n_components_list = args.n_components
  main(base_folder, n_components_list)
