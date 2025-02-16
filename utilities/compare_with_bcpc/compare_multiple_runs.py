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
from typing import Tuple
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
    description="Load and compare pre-processed results"
  )
  parser.add_argument(
    "--result_dirs", 
    help="List of paths to the results folders", 
    type=str,
    nargs="+"
  )
  parser.add_argument(
    "-n", "--n_components", 
    help="Number of components to consider", 
    type=int,
    nargs="+"
  )
  parser.add_argument(
    "--process_all", 
    help="True if all subdirectories of `result_dirs` should be processed", 
    default=False,
    action="store_true"
  )
  args, _ = parser.parse_known_args()
  return args


def add_configuration_info(
    results: pd.DataFrame, tokens: list
  ) -> pd.DataFrame:
  # extract info
  constraints = tokens[0]
  n_RG_iter = int(tokens[1])
  n_LS_iter = int(tokens[2])
  n_elite_sol = int(tokens[3])
  # add
  results["constraints"] = [constraints] * len(results)
  results["n_RG_iter"] = [n_RG_iter] * len(results)
  results["n_LS_iter"] = [n_LS_iter] * len(results)
  results["n_elite_sol"] = [n_elite_sol] * len(results)
  return results


def load_results(complete_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
  s4air_results = pd.read_csv(os.path.join(complete_path, "s4air_results.csv"))
  bcpc_results = pd.read_csv(os.path.join(complete_path, "bcpc_results.csv"))
  return s4air_results, bcpc_results


def plot_comparison(
    baseline_results: pd.DataFrame, 
    method_results: pd.DataFrame, 
    n_components_list: list, 
    config_cols: list,
    ycol: str,
    ylabel: str,
    baseline_name: str = "BCPC",
    method_name: str = "S4AIR",
    gainlabel: str = "Average PCR (%)",
    plot_folder: str = None,
    logy: bool = False
  ):
  config_idx_to_drop = {
    7: [0] + list(range(4,11)), 
    10: [0, 2, 4, 5, 7, 8], 
    15: [0, 2, 4, 5, 7, 8]
  }
  # filter from minimum threshold
  min_threshold = 10
  max_threshold = 75
  baseline_results = baseline_results[
    (
      baseline_results["threshold"] >= min_threshold
    ) & (
      baseline_results["threshold"] <= max_threshold
    )
  ]
  method_results = method_results[
    (
      method_results["threshold"] >= min_threshold
    ) & (
      method_results["threshold"] <= max_threshold
    )
  ]
  # define colors
  colors = list(mcolors.TABLEAU_COLORS.values())[1:] + [
    mcolors.CSS4_COLORS["navy"],
    mcolors.CSS4_COLORS["limegreen"],
    mcolors.CSS4_COLORS["darkred"],
    mcolors.CSS4_COLORS["purple"]
  ]
  # define figure
  ncols = len(n_components_list)
  _, axs = plt.subplots(
    nrows = 2, 
    ncols = ncols, 
    sharex = "col",
    sharey = "row", 
    figsize = (12 * ncols, 8)
  )
  averages = {
    "n_components": [],
    "config": [],
    "config_idx": [],
    "config_idx_to_plot": [],
    "avg": [],
    "min": [],
    "max": []
  }
  fontsize = 18
  # loop over the number of components
  for idx, n_components in enumerate(n_components_list):
    ax0 = axs[0] if ncols == 1 else axs[0][idx]
    ax1 = axs[1] if ncols == 1 else axs[1][idx]
    b_res = baseline_results[baseline_results["n_components"] == n_components]
    all_m_res = method_results[method_results["n_components"] == n_components]
    # compute average and standard deviation for the baseline method
    b_res_avg = b_res.groupby("exp_id").mean(numeric_only = True)[
      ["threshold", ycol]
    ]
    b_res_avg["std"] = b_res.groupby("exp_id").std(numeric_only = True)[ycol]
    b_res_avg["n"] = b_res.groupby("exp_id").count()[ycol]
    # plot the baseline result and confidence intervals
    b_res_avg.plot(
      x = "threshold",
      y = ycol,
      label = baseline_name,
      ax = ax0,
      grid = True,
      fontsize = fontsize,
      marker = ".",
      markersize = 5,
      linewidth = 1,
      color = mcolors.TABLEAU_COLORS["tab:blue"],
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
    # loop over the different configurations
    n_config = 0
    n_config_to_plot = 0
    for config_info, m_res in all_m_res.groupby(config_cols):
      if n_config not in config_idx_to_drop[n_components]:
        # compute average and standard deviation
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
        cl = f"  RndS: {config_info[1]}\n  SLS: {config_info[2]}\n  K: {config_info[3]}"
        m_res_avg.plot(
          x = "threshold",
          y = ycol,
          label = f"{method_name}\n{cl}",
          ax = ax0,
          grid = True,
          fontsize = fontsize,
          marker = ".",
          markersize = 5,
          linewidth = 1,
          color = colors[n_config_to_plot],
          logy = logy
        )
        # confidence intervals
        ax0.fill_between(
          x = m_res_avg["threshold"],
          y1 = m_res_avg[ycol] - 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
          y2 = m_res_avg[ycol] + 0.95 * m_res_avg["std"] / m_res_avg["n"].pow(1./2),
          alpha = 0.3,
          color = colors[n_config_to_plot]
        )
        # # add Xs for unfeasible runs
        # unfeasible = pd.DataFrame()
        # if "cost" in m_res:
        #   unfeasible = m_res[
        #     m_res["cost"] == np.inf # "Unfeasible solution saved"
        #   ].copy(deep = True)
        # if len(unfeasible) > 0:
        #   unfeasible["y"] = [
        #     m_res.drop(unfeasible.index)[ycol].max()
        #   ] * len(unfeasible)
        #   unfeasible.plot.scatter(
        #     x = "threshold",
        #     y = "y",
        #     marker = "*",
        #     s = 50,
        #     color = mcolors.TABLEAU_COLORS["tab:red"],
        #     grid = True,
        #     ax = ax0
        #   )
        # plot gain
        avg_gain.plot(
          x = "threshold",
          y = ycol,
          ax = ax1,
          grid = True,
          fontsize = fontsize,
          marker = ".",
          markersize = 5,
          linewidth = 1,
          color = colors[n_config_to_plot],
          label = None,
          legend = False
        )
        ax1.axhline(
          y = avg_gain[avg_gain[ycol].abs() != np.inf][ycol].mean(),
          linewidth = 2,
          color = colors[n_config_to_plot]
        )
        # save average for barplot
        averages["n_components"].append(n_components)
        averages["config"].append(f"{method_name}\n{cl}")
        averages["config_idx"].append(n_config)
        averages["config_idx_to_plot"].append(n_config_to_plot)
        averages["avg"].append(
          avg_gain[avg_gain[ycol].abs() != np.inf][ycol].mean()
        )
        averages["min"].append(
          avg_gain[avg_gain[ycol].abs() != np.inf][ycol].min()
        )
        averages["max"].append(
          avg_gain[avg_gain[ycol].abs() != np.inf][ycol].max()
        )
        # if len(unfeasible) > 0:
        #   unfeasible["y"] = [0] * len(unfeasible)
        #   unfeasible.plot.scatter(
        #     x = "threshold",
        #     y = "y",
        #     marker = "*",
        #     s = 50,
        #     color = mcolors.TABLEAU_COLORS["tab:red"],
        #     grid = True,
        #     ax = ax1
        #   )
        n_config_to_plot += 1
      n_config += 1
    # add horizontal line in zero
    ax1.axhline(
      y = 0,
      linestyle = "dashed",
      color = "k"
    )
    # add axis info
    ax0.set_title(f"{n_components} components", fontsize = fontsize)
    ax0.legend(ncol = 2)
    ax1.set_xlabel("Global constraint threshold", fontsize = fontsize)
  if ncols > 1:
    axs[0][0].set_ylabel(ylabel, fontsize = fontsize)
    axs[1][0].set_ylabel(gainlabel, fontsize = fontsize)
    for idx in range(ncols-1):
      axs[0,idx].get_legend().remove()
    axs[0,-1].legend(
      loc = "center left", bbox_to_anchor = (1, -0.1), fontsize = fontsize
    )
  else:
    axs[0].set_ylabel(ylabel, fontsize = fontsize)
    axs[1].set_ylabel(gainlabel, fontsize = fontsize)
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
  # plot averages as bars
  averages = pd.DataFrame(averages)
  hatches = list('-./')
  _, axs = plt.subplots(
    nrows = 1, 
    ncols = ncols, 
    sharex = "col",
    sharey = "row", 
    figsize = (12 * ncols, 8)
  )
  idx = 0
  for n_components, avgs in averages.groupby("n_components"):
    ax = axs if ncols == 1 else axs[idx]
    for i, c in enumerate(["avg", "min", "max"]):
      offset = 0.3 - 0.3 * i# if idx == 0
      ax.bar(
        x = avgs.index - offset, 
        height = avgs[c], 
        width = 0.3, 
        hatch = hatches[i], 
        label = c,
        color = [colors[cidx] for cidx in avgs["config_idx_to_plot"]],
        edgecolor = "k",
        alpha = 0.8
      )
    ax.axhline(
      y = 0,
      linestyle = "dashed",
      color = "k"
    )
    ax.set_title(f"{n_components} components", fontsize = fontsize)
    ax.legend(fontsize = fontsize)
    ax.grid()
    idx += 1
  axs[0].set_ylabel(gainlabel, fontsize = fontsize)
  if plot_folder is not None:
    plt.savefig(
      os.path.join(plot_folder, f"{ycol}_comparison_avg.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()
  # save averages
  if plot_folder is not None:
    averages.to_csv(
      os.path.join(plot_folder, f"{ycol}_averages.csv"), index = False
    )


def main(result_dirs: list, n_components: list, process_all: bool):
  base_folder = None
  if process_all:
    base_folder = result_dirs[0]
    result_dirs = [
      os.path.join(base_folder, d) for d in os.listdir(base_folder) \
        if os.path.isdir(os.path.join(base_folder, d))
    ]
  # loop over all result folders
  all_s4air_results = pd.DataFrame()
  bcpc_results = None
  for result_dir in result_dirs:
    _, dirname = os.path.split(result_dir)
    # get configuration info
    tokens = parse("{}Constraints_RG{}_LS{}_{}sol", dirname)
    if tokens is not None:
      # load results
      s4air_results, bcpc_results = load_results(result_dir)
      # add configuration info and merge
      s4air_results = add_configuration_info(s4air_results, tokens)
      all_s4air_results = pd.concat(
        [all_s4air_results, s4air_results], ignore_index = True
      )
  # plot cost comparison
  plot_folder = "."
  if process_all:
    plot_folder = base_folder
  plot_comparison(
    baseline_results = bcpc_results,
    method_results = all_s4air_results,
    n_components_list = [7, 10, 15],
    config_cols = [
      "constraints",
      "n_RG_iter",
      "n_LS_iter",
      "n_elite_sol"
    ],
    ycol = "cost",
    ylabel = "Cost",
    plot_folder = plot_folder
  )
  # plot runtime comparison
  plot_comparison(
    baseline_results = bcpc_results,
    method_results = all_s4air_results,
    n_components_list = [7, 10, 15],
    config_cols = [
      "constraints",
      "n_RG_iter",
      "n_LS_iter",
      "n_elite_sol"
    ],
    ycol = "exec_time",
    ylabel = "Runtime (s)",
    plot_folder = plot_folder,
    gainlabel = "Average time reduction (%)",
    logy = True
  )


if __name__ == "__main__":
  result_dirs = ["/Users/federicafilippini/Documents/TEMP/S4AIRvsBCPC/20250124/postprocessing"]
  n_components = [7, 10, 15]
  process_all = True
  main(result_dirs, n_components, process_all)
