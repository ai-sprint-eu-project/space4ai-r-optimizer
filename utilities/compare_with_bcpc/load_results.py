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
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import numpy as np
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Load BCPC results"
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


def load_bcpc_result(
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


def load_s4air_result(
    instance_folder: str, n_components: int, instance_id: int
  ) -> pd.DataFrame:
  results = {}
  # loop over all files
  for filename in os.listdir(instance_folder):
    if filename.startswith("s4air_solution_"):
      # get the constraint threshold
      tokens = filename.split("_")
      key = "_".join(tokens[1:-1])
      threshold = int(tokens[-1].split(".")[0])
      if threshold not in results:
        results[threshold] = {}
      # load result
      res_dict = {}
      with open(os.path.join(instance_folder, filename), "r") as istream:
        res_dict = json.load(istream)
      # save relevant information
      results[threshold]["cost"] = res_dict.get("total_cost", np.inf)
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


def load_all_results(
    base_folder: str, n_components_list: list, method: str
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
        instance_results = None
        if method == "bcpc":
          instance_results = load_bcpc_result(
            instance_folder, n_components, instance_id
          )
        elif method == "s4air":
          instance_results = load_s4air_result(
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


def plot_method_results(
    all_results: pd.DataFrame, 
    n_components_list: list, 
    xcol: str, 
    ycol: str,
    xlabel: str,
    ylabel: str,
    method: str,
    plot_folder: str = None,
    logy: bool = False
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
        y = ycol,
        label = f"Ins{instance_id}",
        ax = axs[idx],
        grid = True,
        fontsize = 14,
        marker = ".",
        markersize = 5,
        linewidth = 1,
        alpha = 0.7,
        logy = logy
      )
    # plot average
    results.groupby("exp_id").mean(numeric_only = True).plot(
      x = xcol,
      y = ycol,
      label = "average",
      color = "k",
      ax = axs[idx],
      grid = True,
      fontsize = 14,
      linewidth = 3,
      logy = logy
    )
    # add axis info
    axs[idx].set_xlabel(xlabel, fontsize = 14)
    axs[idx].set_title(f"{n_components} components", fontsize = 14)
  axs[0].set_ylabel(ylabel, fontsize = 14)
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"{method}_{ycol}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def plot_comparison(
    baseline_results: pd.DataFrame, 
    method_results: pd.DataFrame, 
    n_components_list: list, 
    plot_folder: str = None
  ):
  _, axs = plt.subplots(
    nrows = 1, ncols = len(n_components_list), sharey = True, figsize = (25, 5)
  )
  for idx, n_components in enumerate(n_components_list):
    b_res = baseline_results[baseline_results["n_components"] == n_components]
    m_res = method_results[method_results["n_components"] == n_components]
    # compute average and standard deviation
    b_res_avg = b_res.groupby("exp_id").mean(numeric_only = True)[
      ["threshold", "cost"]
    ]
    b_res_avg["std"] = b_res.groupby("exp_id").std(numeric_only = True)["cost"]
    b_res_avg["n"] = b_res.groupby("exp_id").count()["cost"]
    m_res_avg = m_res.groupby("exp_id").mean(numeric_only = True)[
      ["threshold", "cost"]
    ]
    m_res_avg["std"] = m_res.groupby("exp_id").std(numeric_only = True)["cost"]
    m_res_avg["n"] = m_res.groupby("exp_id").count()["cost"]
    # plot
    b_res_avg.plot(
      x = "threshold",
      y = "cost",
      label = "BCPC",
      ax = axs[idx],
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:blue"]
    )
    m_res_avg.plot(
      x = "threshold",
      y = "cost",
      label = "S4AIR",
      ax = axs[idx],
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:orange"]
    )
    # confidence intervals
    axs[idx].fill_between(
      x = b_res_avg["threshold"],
      y1 = b_res_avg["cost"] - 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      y2 = b_res_avg["cost"] + 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      alpha = 0.4,
      color = mcolors.TABLEAU_COLORS["tab:blue"]
    )
    axs[idx].fill_between(
      x = m_res_avg["threshold"],
      y1 = m_res_avg["cost"] - 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      y2 = m_res_avg["cost"] + 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      alpha = 0.3,
      color = mcolors.TABLEAU_COLORS["tab:orange"]
    )
    # add axis info
    axs[idx].set_xlabel("Global constraint threshold", fontsize = 14)
    axs[idx].set_title(f"{n_components} components", fontsize = 14)
  axs[0].set_ylabel("Cost", fontsize = 14)
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, "comparison.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def main(base_folder: str, n_components_list: list):
  # load BCPC and SPACE4AI-R results
  all_bcpc_results = load_all_results(base_folder, n_components_list, "bcpc")
  all_bcpc_results["method"] = ["BCPC"] * len(all_bcpc_results)
  all_s4air_results = load_all_results(base_folder, n_components_list, "s4air")
  all_s4air_results["method"] = ["SPACE4AI-R"] * len(all_s4air_results)
  # normalize the threshold values
  all_bcpc_results = normalize_threshold(all_bcpc_results)
  all_bcpc_results.to_csv(os.path.join(base_folder, "bcpc_results.csv"))
  all_s4air_results = normalize_threshold(all_s4air_results)
  all_s4air_results.to_csv(os.path.join(base_folder, "s4air_results.csv"))
  # plot
  plot_method_results(
    all_bcpc_results, 
    n_components_list, 
    "threshold", 
    "cost",
    "Global constraint threshold",
    "Cost",
    "bcpc",
    base_folder
  )
  plot_method_results(
    all_s4air_results, 
    n_components_list, 
    "threshold", 
    "cost",
    "Global constraint threshold",
    "Cost",
    "s4air",
    base_folder
  )
  # plot comparison
  plot_comparison(
    all_bcpc_results, 
    all_s4air_results, 
    n_components_list, 
    base_folder
  )
  # plot method runtime
  plot_method_results(
    all_bcpc_results, 
    n_components_list, 
    "threshold", 
    "exec_time",
    "Global constraint threshold",
    "Method runtime (s)",
    "bcpc",
    plot_folder = base_folder,
    logy = True
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  n_components_list = args.n_components
  main(base_folder, n_components_list)
