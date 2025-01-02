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
from datetime import datetime
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
          instance_logs = parse_s4air_logs(
            instance_folder, n_components, instance_id
          )
          if len(instance_results) > 0 and len(instance_logs) > 0:
            instance_results = instance_results.join(
              instance_logs[["exec_time", "status"]]
            )
        # merge results
        all_results = pd.concat([all_results, instance_results])
  return all_results


def parse_s4air_logs(
    instance_folder: str, n_components: int, instance_id: int
  ) -> pd.DataFrame:
  exec_times = {}
  # loop over all files
  for filename in os.listdir(instance_folder):
    if filename.startswith("s4air_") and filename.endswith(".log"):
      # get the constraint threshold
      tokens = filename.split("_")
      threshold = int(tokens[-1].split(".")[0])
      if threshold not in exec_times:
        exec_times[threshold] = {}
      # parse log
      exec_time = np.inf
      status = "ERROR"
      with open(os.path.join(instance_folder, filename), "r") as istream:
        lines = istream.readlines()
        # start time
        start, _ = parse(
          "{}	[Info]	****** READING CONFIGURATION FILE: {} ... ******\n",
          lines[0]
        )
        start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        # finish time
        end, status, _ = parse("{}	[Info]	{} at: {}\n", lines[-2])
        end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        # duration
        exec_time = (end - start).total_seconds()
      # save relevant information
      exec_times[threshold]["exec_time"] = float(exec_time)
      exec_times[threshold]["status"] = status
  # build dataframe
  exec_times = pd.DataFrame(exec_times).transpose().sort_index()
  for c in exec_times.columns:
    if "status" not in c:
      exec_times[c] = exec_times[c].astype("float")
  # rename index column
  exec_times.index.name = "original_threshold"
  # add number of components and instance id
  exec_times["n_components"] = [n_components] * len(exec_times)
  exec_times["instance"] = [instance_id] * len(exec_times)
  return exec_times


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
  ncols = len(n_components_list)
  _, axs = plt.subplots(
    nrows = 1, 
    ncols = ncols, 
    sharey = True, 
    figsize = (8 * ncols, 5)
  )
  for idx, n_components in enumerate(n_components_list):
    results = all_results[all_results["n_components"] == n_components]
    ax = axs if ncols == 1 else axs[idx]
    # plot instance results
    for instance_id, data in results.groupby("instance"):
      data.plot(
        x = xcol,
        y = ycol,
        label = f"Ins{instance_id}",
        ax = ax,
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
      ax = ax,
      grid = True,
      fontsize = 14,
      linewidth = 3,
      logy = logy
    )
    # add axis info
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_title(f"{n_components} components", fontsize = 14)
  if ncols > 1:
    axs[0].set_ylabel(ylabel, fontsize = 14)
  else:
    axs.set_ylabel(ylabel, fontsize = 14)
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
    ycol: str,
    ylabel: str,
    plot_folder: str = None,
    logy: bool = False
  ):
  ncols = len(n_components_list)
  _, axs = plt.subplots(
    nrows = 1, ncols = ncols, sharey = True, figsize = (8 * ncols, 5)
  )
  for idx, n_components in enumerate(n_components_list):
    ax = axs if ncols == 1 else axs[idx]
    b_res = baseline_results[baseline_results["n_components"] == n_components]
    m_res = method_results[method_results["n_components"] == n_components]
    # compute average and standard deviation
    b_res_avg = b_res.groupby("exp_id").mean(numeric_only = True)[
      ["threshold", ycol]
    ]
    b_res_avg["std"] = b_res.groupby("exp_id").std(numeric_only = True)[ycol]
    b_res_avg["n"] = b_res.groupby("exp_id").count()[ycol]
    m_res_avg = m_res.groupby("exp_id").mean(numeric_only = True)[
      ["threshold", ycol]
    ]
    m_res_avg["std"] = m_res.groupby("exp_id").std(numeric_only = True)[ycol]
    m_res_avg["n"] = m_res.groupby("exp_id").count()[ycol]
    # plot
    b_res_avg.plot(
      x = "threshold",
      y = ycol,
      label = "BCPC",
      ax = ax,
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:blue"],
      logy = logy
    )
    m_res_avg.plot(
      x = "threshold",
      y = ycol,
      label = "S4AIR",
      ax = ax,
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:orange"],
      logy = logy
    )
    # confidence intervals
    ax.fill_between(
      x = b_res_avg["threshold"],
      y1 = b_res_avg[ycol] - 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      y2 = b_res_avg[ycol] + 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      alpha = 0.4,
      color = mcolors.TABLEAU_COLORS["tab:blue"]
    )
    ax.fill_between(
      x = m_res_avg["threshold"],
      y1 = m_res_avg[ycol] - 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      y2 = m_res_avg[ycol] + 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      alpha = 0.3,
      color = mcolors.TABLEAU_COLORS["tab:orange"]
    )
    # add axis info
    ax.set_xlabel("Global constraint threshold", fontsize = 14)
    ax.set_title(f"{n_components} components", fontsize = 14)
  if ncols > 1:
    axs[0].set_ylabel(ylabel, fontsize = 14)
  else:
    axs.set_ylabel(ylabel, fontsize = 14)
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"{ycol}_comparison.png"),
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
    "cost",
    "Cost",
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
  plot_method_results(
    all_s4air_results, 
    n_components_list, 
    "threshold", 
    "exec_time",
    "Global constraint threshold",
    "Method runtime (s)",
    "s4air",
    plot_folder = base_folder,
    logy = True
  )
  # plot runtime comparison
  plot_comparison(
    all_bcpc_results, 
    all_s4air_results, 
    n_components_list, 
    "exec_time",
    "Runtime (s)",
    plot_folder = base_folder,
    logy = True
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  n_components_list = args.n_components
  main(base_folder, n_components_list)
