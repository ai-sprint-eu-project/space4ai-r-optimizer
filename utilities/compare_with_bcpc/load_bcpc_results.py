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
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import numpy as np
import argparse
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


def load_instance_result(
    instance_folder: str, n_components: int, instance_id: int
  ) -> pd.DataFrame:
  results = {}
  # loop over all files
  for filename in os.listdir(instance_folder):
    if filename.startswith("paper_"):
      # get the constraint threshold
      tokens = filename.split("_")
      key = "_".join(tokens[1:-1])
      threshold = int(tokens[-1].split(".")[0])
      if threshold not in results:
        results[threshold] = {}
      # load result
      key_data = np.load(
        os.path.join(instance_folder, filename), allow_pickle = True
      )
      results[threshold][key] = key_data
  # build dataframe
  results = pd.DataFrame(results).transpose().sort_index()
  for c in results.columns:
    if "mem" not in c:
      results[c] = results[c].astype("float")
    else:
      results[c] = [
        {k: int(v) for k, v in d.item().items()} for d in results[c]
      ]
  # rename index column
  results.index.name = "original_threshold"
  # add number of components and instance id
  results["n_components"] = [n_components] * len(results)
  results["instance"] = [instance_id] * len(results)
  return results


def load_bcpc_results(
    base_folder: str, n_components_list: list
  ) -> pd.DataFrame:
  all_results = pd.DataFrame()
  # loop over scenarios (number of components)
  for n_components in n_components_list:
    scenario = f"{n_components}Components"
    print(f"Processing scenario {scenario}")
    scenario_folder = os.path.join(base_folder, scenario)
    # loop over instances
    for foldername in os.listdir(scenario_folder):
      instance_id = parse("Ins{}", foldername)
      if instance_id is not None:
        instance_id = int(instance_id[0])
        print(f"  processing instance {instance_id}")
        # load instance results
        instance_folder = os.path.join(scenario_folder, foldername)
        instance_results = load_instance_result(
          instance_folder, n_components, instance_id
        )
        # merge results
        all_results = pd.concat([all_results, instance_results])
  return all_results


def normalize_threshold(all_data: pd.DataFrame) -> pd.DataFrame:
  all_results = all_data.copy(deep = True)
  all_results["original_threshold"] = all_results.index
  all_results.index = list(range(len(all_results)))
  # normalize the constraint value
  all_results["threshold"] = [-1.0] * len(all_results)
  all_results["exp_id"] = [None] * len(all_results)
  for _, data in all_results.groupby(["n_components", "instance"]):
    tmim = data["original_threshold"].min()
    tmax = data["original_threshold"].max()
    all_results.loc[data.index, "threshold"] = [
      rescale(val, tmim, tmax, 0, 100) for val in data["original_threshold"]
    ]
    all_results.loc[data.index, "exp_id"] = list(range(len(data)))
  return all_results


def plot_bcpc_results(
    all_results: pd.DataFrame, 
    n_components_list: list, 
    xcol: str, 
    xlabel: str,
    plot_folder: str = None
  ):
  _, axs = plt.subplots(
    nrows = 1, ncols = len(n_components_list), sharey = True, figsize = (25, 5)
  )
  for idx, n_components in enumerate(n_components_list):
    results = all_results[all_results["n_components"] == n_components]
    # plot instance results
    for instance_id, data in results.groupby("instance"):
      data.plot(
        x = xcol,
        y = "cost",
        label = f"Ins{instance_id}",
        ax = axs[idx],
        grid = True,
        fontsize = 14,
        marker = ".",
        markersize = 5,
        linewidth = 1,
        alpha = 0.7
      )
    # plot average
    results.groupby("exp_id").mean(numeric_only = True).plot(
      x = xcol,
      y = "cost",
      label = "average",
      color = "k",
      ax = axs[idx],
      grid = True,
      fontsize = 14,
      linewidth = 3
    )
    # add axis info
    axs[idx].set_xlabel(xlabel, fontsize = 14)
    axs[idx].set_title(f"{n_components} components", fontsize = 14)
  axs[0].set_ylabel("Cost", fontsize = 14)
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, "bcpc_results.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def main(base_folder: str, n_components_list: list):
  # load BCPC results
  all_results = load_bcpc_results(base_folder, n_components_list)
  # normalize the threshold values
  all_results = normalize_threshold(all_results)
  all_results.to_csv(os.path.join(base_folder, "bcpc_results.csv"))
  # plot
  plot_bcpc_results(
    all_results, 
    n_components_list, 
    "threshold", 
    "Global constraint threshold",
    base_folder
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  n_components_list = args.n_components
  main(base_folder, n_components_list)
