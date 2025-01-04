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
    description="Load BCPC and SPACE4AI-R results"
  )
  parser.add_argument(
    "--application_dir", 
    help="Path to the base folder with SPACE4AI-R results", 
    type=str
  )
  parser.add_argument(
    "--bcpc_dir", 
    help="Path to the folder with BCPC results", 
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
              instance_logs.drop(
                [c for c in instance_logs.columns if c in instance_results.columns],
                axis = "columns"
              )
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
      spaces = 0
      nfound = 0
      with open(os.path.join(instance_folder, filename), "r") as istream:
        lines = istream.readlines()
        for idx, line in enumerate(lines):
          if idx == 0:
            # start time
            start, _ = parse(
              "{}	[Info]	****** READING CONFIGURATION FILE: {} ... ******\n",
              lines[0]
            )
            start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
          elif idx == len(lines) - 2 and "saved" in line:
            # finish time
            end, status, _ = parse("{}	[Info]	{} at: {}\n", lines[-2])
            end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
            # duration
            exec_time = (end - start).total_seconds()
          elif "Elite container initialized" in line:
            # container spaces
            _, spaces = parse(
              "{}	[Info]	Elite container initialized with {} spaces\n", line
            )
          elif "Number of top feasible" in line:
            # number of feasible solutions found
            _, nfound = parse(
              "{}	[Info]	Number of top feasible solutions found: {}\n", line
            )
      # save relevant information
      exec_times[threshold]["exec_time"] = float(exec_time)
      exec_times[threshold]["status"] = status
      exec_times[threshold]["spaces"] = int(spaces)
      exec_times[threshold]["nfound"] = int(nfound)
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


def normalize_threshold(
    all_data: pd.DataFrame, thresholds: dict
  ) -> pd.DataFrame:
  all_results = all_data.copy(deep = True)
  all_results["original_threshold"] = all_results.index
  all_results.index = list(range(len(all_results)))
  # normalize the constraint value
  all_results["threshold"] = [-1.0] * len(all_results)
  all_results["exp_id"] = [None] * len(all_results)
  for key, data in all_results.groupby(["n_components", "instance"]):
    c = f"{int(key[0])}Components"
    i = f"Ins{int(key[1])}"
    df = pd.DataFrame(thresholds[c][i]).sort_values("rescaled")
    mapping = {
      k: (v, j) for k,v,j in zip(
        df["original"], df["rescaled"], range(len(df["rescaled"]))
      )
    }
    all_results.loc[data.index, "threshold"] = [
      mapping[val][0] for val in data["original_threshold"]
    ]
    all_results.loc[data.index, "exp_id"] = [
      mapping[val][1] for val in data["original_threshold"]
    ]
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
    # add Xs for unfeasible runs
    unfeasible = results[
      results["cost"] == np.inf # "Unfeasible solution saved"
    ].copy(deep = True)
    if len(unfeasible) > 0:
      unfeasible["y"] = [
        results.drop(unfeasible.index)[ycol].max()
      ] * len(unfeasible)
      unfeasible.plot.scatter(
        x = "threshold",
        y = "y",
        marker = "*",
        s = 50,
        color = mcolors.TABLEAU_COLORS["tab:red"],
        grid = True,
        ax = ax
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
    baseline_name: str = "BCPC",
    method_name: str = "S4AIR",
    gainlabel: str = "Average PCR (%)",
    plot_folder: str = None,
    logy: bool = False
  ):
  ncols = len(n_components_list)
  _, axs = plt.subplots(
    nrows = 2, 
    ncols = ncols, 
    sharex = "col",
    sharey = "row", 
    figsize = (8 * ncols, 8)
  )
  for idx, n_components in enumerate(n_components_list):
    ax0 = axs[0] if ncols == 1 else axs[0][idx]
    ax1 = axs[1] if ncols == 1 else axs[1][idx]
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
    # compute gain
    avg_gain = pd.DataFrame({
      ycol: (b_res_avg[ycol] - m_res_avg[ycol]) / b_res_avg[ycol] * 100,
      "threshold": b_res_avg["threshold"]
    })
    # plot
    b_res_avg.plot(
      x = "threshold",
      y = ycol,
      label = baseline_name,
      ax = ax0,
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
      label = method_name,
      ax = ax0,
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:orange"],
      logy = logy
    )
    # confidence intervals
    ax0.fill_between(
      x = b_res_avg["threshold"],
      y1 = b_res_avg[ycol] - 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      y2 = b_res_avg[ycol] + 0.95 * b_res_avg["std"] / b_res_avg["n"].pow(1./2),
      alpha = 0.4,
      color = mcolors.TABLEAU_COLORS["tab:blue"]
    )
    ax0.fill_between(
      x = m_res_avg["threshold"],
      y1 = m_res_avg[ycol] - 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      y2 = m_res_avg[ycol] + 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
      alpha = 0.3,
      color = mcolors.TABLEAU_COLORS["tab:orange"]
    )
    # add Xs for unfeasible runs
    unfeasible = pd.DataFrame()
    if "cost" in m_res:
      unfeasible = m_res[
        m_res["cost"] == np.inf # "Unfeasible solution saved"
      ].copy(deep = True)
    if len(unfeasible) > 0:
      unfeasible["y"] = [
        m_res.drop(unfeasible.index)[ycol].max()
      ] * len(unfeasible)
      unfeasible.plot.scatter(
        x = "threshold",
        y = "y",
        marker = "*",
        s = 50,
        color = mcolors.TABLEAU_COLORS["tab:red"],
        grid = True,
        ax = ax0
      )
    # plot gain
    avg_gain.plot(
      x = "threshold",
      y = ycol,
      ax = ax1,
      grid = True,
      fontsize = 14,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:green"],
      label = None,
      legend = False
    )
    ax1.axhline(
      y = 0,
      linestyle = "dashed",
      color = "k"
    )
    ax1.axhline(
      y = avg_gain[avg_gain[ycol].abs() != np.inf][ycol].mean(),
      linewidth = 2,
      color = mcolors.TABLEAU_COLORS["tab:red"]
    )
    if len(unfeasible) > 0:
      unfeasible["y"] = [0] * len(unfeasible)
      unfeasible.plot.scatter(
        x = "threshold",
        y = "y",
        marker = "*",
        s = 50,
        color = mcolors.TABLEAU_COLORS["tab:red"],
        grid = True,
        ax = ax1
      )
    # add axis info
    ax0.set_title(f"{n_components} components", fontsize = 14)
    ax1.set_xlabel("Global constraint threshold", fontsize = 14)
  if ncols > 1:
    axs[0][0].set_ylabel(ylabel, fontsize = 14)
    axs[1][0].set_ylabel(gainlabel, fontsize = 14)
  else:
    axs[0].set_ylabel(ylabel, fontsize = 14)
    axs[1].set_ylabel(gainlabel, fontsize = 14)
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


def main(s4air_folder: str, bcpc_folder: str, n_components_list: list):
  base_folder, output_foldername = os.path.split(s4air_folder)
  # generate folder to save results
  output_folder = os.path.join(
    base_folder, 
    "postprocessing",
    output_foldername.removeprefix("output_")
  )
  os.makedirs(output_folder, exist_ok = True)
  # load BCPC and SPACE4AI-R results
  all_bcpc_results = load_all_results(
    bcpc_folder, n_components_list, "bcpc"
  )
  all_bcpc_results["method"] = ["BCPC"] * len(all_bcpc_results)
  all_s4air_results = load_all_results(
    s4air_folder, n_components_list, "s4air"
  )
  all_s4air_results["method"] = ["SPACE4AI-R"] * len(all_s4air_results)
  # load normalized threshold values
  thresholds = {}
  with open(os.path.join(bcpc_folder, "thresholds.json"), "r") as ist:
    thresholds = json.load(ist)
  # normalize the threshold values
  all_bcpc_results = normalize_threshold(all_bcpc_results, thresholds)
  all_bcpc_results.to_csv(os.path.join(output_folder, "bcpc_results.csv"))
  all_s4air_results = normalize_threshold(all_s4air_results, thresholds)
  all_s4air_results.to_csv(os.path.join(output_folder, "s4air_results.csv"))
  # plot
  plot_method_results(
    all_bcpc_results, 
    n_components_list, 
    "threshold", 
    "cost",
    "Global constraint threshold",
    "Cost",
    "bcpc",
    output_folder
  )
  plot_method_results(
    all_s4air_results, 
    n_components_list, 
    "threshold", 
    "cost",
    "Global constraint threshold",
    "Cost",
    "s4air",
    output_folder
  )
  # plot comparison
  plot_comparison(
    all_bcpc_results, 
    all_s4air_results, 
    n_components_list, 
    "cost",
    "Cost",
    plot_folder = output_folder
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
    plot_folder = output_folder,
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
    plot_folder = output_folder,
    logy = True
  )
  # plot runtime comparison
  plot_comparison(
    all_bcpc_results, 
    all_s4air_results, 
    n_components_list, 
    "exec_time",
    "Runtime (s)",
    gainlabel = "Average time reduction (%)",
    plot_folder = output_folder,
    logy = True
  )
  # plot n. found solutions
  plot_comparison(
    all_s4air_results[
      ["spaces", "threshold", "n_components", "exp_id"]
    ].rename({"spaces": "n_solutions"}, axis = "columns"), 
    all_s4air_results[
      ["nfound", "threshold", "n_components", "exp_id"]
    ].rename({"nfound": "n_solutions"}, axis = "columns"), 
    n_components_list, 
    "n_solutions",
    "# elite solutions",
    baseline_name = "spaces",
    method_name = "found",
    gainlabel = "Residual space (%)",
    plot_folder = output_folder
  )


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.application_dir
  bcpc_folder = args.bcpc_dir
  n_components_list = args.n_components
  main(base_folder, bcpc_folder, n_components_list)
