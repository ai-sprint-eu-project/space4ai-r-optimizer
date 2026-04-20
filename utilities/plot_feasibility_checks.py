"""
Copyright 2026 AI-SPRINT

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
from plot_results import barplot

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import ast
import os


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Results Postprocessing (feasibility checks)"
    )
    parser.add_argument(
      "--results_dir", 
      help="Path to the results directory", 
      type=str
    )
    parser.add_argument(
      "--heuristics", 
      help="List of heuristics to consider (available: s4air, uheur)", 
      nargs="+",
      default=["s4air"]
    )
    parser.add_argument(
      "--target", 
      help="Target method", 
      type=str,
      default="s4air"
    )
    parser.add_argument(
      "--skip_instance", 
      help="True if instance-specific plots should not be generated", 
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--skip_scenario", 
      help="True if scenario-specific plots should not be generated", 
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--skip_workload", 
      help="True if workload-specific plots should not be generated", 
      default=False,
      action="store_true"
    )
    args, _ = parser.parse_known_args()
    return args


def evaluate_instance(
        instance_dir: str, 
        heuristics: list, 
        target: str,
        skip: dict
    ) -> Tuple[dict, dict, dict]:
    # create directory to store figures
    plot_dir = os.path.join(instance_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # load feasibility reports
    full_report = {}
    for fname in os.listdir(instance_dir):
        if fname.startswith("trace_feasibility-"):
            tokens = fname.split("-")[1:]
            rule = tokens[0]
            workload_noise = float(
                tokens[1].split("_")[-1].replace(".txt", "")
            )
            if workload_noise not in full_report:
                full_report[workload_noise] = {}
            # load
            lines = []
            with open(os.path.join(instance_dir, fname), "r") as ist:
                lines = ist.readlines()
            # decode lines and generate report
            report = pd.DataFrame()
            for line in lines:
                linedict = ast.literal_eval(line)
                linedf = pd.DataFrame()
                for heur in heuristics:
                    if f"{heur}_feasible" in linedict:
                        d = pd.DataFrame(
                            linedict[f"{heur}_feasible"], index = [heur]
                        )
                        d["bandwidth"] = linedict["bandwidth"]
                        d["min_bandwidth_interval"] = linedict[
                            "min_bandwidth_interval"
                        ]
                        d["workload"] = linedict["workload"]
                        d["max_workload_interval"] = linedict[
                            "max_workload_interval"
                        ]
                        d["workload_noise"] = workload_noise
                        if os.path.exists(
                                os.path.join(instance_dir, heur, rule)
                            ):
                            d["rule"] = rule
                        else:
                            d["rule"] = "None"
                        linedf = pd.concat([linedf, d])
                linedf.reset_index(inplace = True, names = "method")
                report = pd.concat([report, linedf], ignore_index = True)
            for heur, heurdata in report.groupby("method"):
                if heur not in full_report[workload_noise]:
                    full_report[workload_noise][heur] = heurdata
                else:
                    if heurdata["rule"].count() > 0:
                        full_report[workload_noise][heur] = pd.concat(
                            [full_report[workload_noise][heur], heurdata],
                            ignore_index = True
                        )
            # plot feasibility
            if not skip["instance"]:
                colors = list(mcolors.TABLEAU_COLORS.values())
                _, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,6))
                for idx, (heur, heurdata) in enumerate(
                        report.groupby("method")
                    ):
                    heurdata["time"] = range(len(heurdata))
                    if heur == target:
                        # -- base workload
                        heurdata.plot.scatter(
                            x = "time",
                            y = "workload",
                            marker = "*",
                            c = "k",
                            s = 30,
                            grid = True,
                            ax = axs[0], 
                            fontsize = 14
                        )
                        heurdata.plot(
                            x = "time",
                            y = "max_workload_interval",
                            color = "k",
                            linestyle = "dashed",
                            linewidth = 2,
                            grid = True,
                            ax = axs[0], 
                            fontsize = 14
                        )
                        # -- base bandwidth
                        heurdata["time"] = range(len(heurdata))
                        heurdata.plot.scatter(
                            x = "time",
                            y = "bandwidth",
                            marker = "*",
                            c = "k",
                            s = 30,
                            grid = True,
                            ax = axs[1], 
                            fontsize = 14
                        )
                        heurdata.plot(
                            x = "time",
                            y = "min_bandwidth_interval",
                            color = "k",
                            linestyle = "dashed",
                            linewidth = 2,
                            grid = True,
                            ax = axs[1], 
                            fontsize = 14
                        )
                    # -- heur workload
                    heurdata.plot(
                        x = "time",
                        y = "max_workload",
                        color = colors[idx],
                        # linestyle = "dashed",
                        linewidth = 3,
                        alpha = 0.7,
                        grid = True,
                        ax = axs[0],
                        label = heur, 
                        fontsize = 14
                    )
                    # -- heur bandwidth
                    heurdata.plot(
                        x = "time",
                        y = "min_bandwidth",
                        color = colors[idx],
                        # linestyle = "dashed",
                        linewidth = 3,
                        alpha = 0.7,
                        grid = True,
                        ax = axs[1],
                        label = heur, 
                        fontsize = 14
                    )
                axs[0].set_xlabel("time [min]", fontsize = 14)
                axs[0].set_ylabel("workload [req/s]", fontsize = 14)
                axs[1].set_xlabel("time [min]", fontsize = 14)
                axs[1].set_ylabel("bandwidth [Mbps]", fontsize = 14)
                axs[0].legend(fontsize = 14)
                axs[1].legend(fontsize = 14)
                plt.savefig(
                    os.path.join(
                        plot_dir, 
                        f"bandwidthlimits-{rule}-{workload_noise}.png"
                    ),
                    dpi = 300,
                    format = "png",
                    bbox_inches = "tight"
                )
                plt.close()
    # define violations
    workload_boundaries = {}
    n_violations = {}
    for wnb, results in full_report.items():
        workload_boundaries[wnb] = {
            "colnames": None,
            "data": pd.DataFrame()
        }
        n_violations[wnb] = pd.DataFrame()
        for heur, heurdata in results.items():
            heurdata["b_f"] = (
                heurdata["min_bandwidth"] >= heurdata["min_bandwidth_interval"]
            )
            colnames = ["b_f"]
            for wn in np.arange(start = 0.05, stop = wnb, step = 0.05):
                heurdata[f"w_f-{wn}"] = (
                    heurdata["max_workload"] >= (
                        heurdata["workload"] * (1 + wn)
                    )
                )
                colnames.append(f"w_f-{wn}")
                heurdata[f"combined_f-{wn}"] = (
                    heurdata["b_f"] & heurdata[f"w_f-{wn}"]
                )
                colnames.append(f"combined_f-{wn}")
            heurdata[f"w_f-{wnb}"] = (
                heurdata["max_workload"] >= heurdata["max_workload_interval"]
            )
            colnames.append(f"w_f-{wnb}")
            heurdata[f"combined_f-{wnb}"] = (
                heurdata["b_f"] & heurdata[f"w_f-{wnb}"]
            )
            colnames.append(f"combined_f-{wnb}")
            if workload_boundaries[wnb]["colnames"] is None:
                workload_boundaries[wnb]["colnames"] = colnames
            workload_boundaries[wnb]["data"] = pd.concat(
                [workload_boundaries[wnb]["data"], heurdata], 
                ignore_index = True
            )
            # compute number of violations per instance
            for rule, ruledata in heurdata.groupby("rule"):
                heurv = ruledata[colnames].copy(deep = True)
                heurv = pd.concat([
                    pd.DataFrame(heurv.sum(), columns = ["n_feasible"]).T,
                    pd.DataFrame(
                        (len(heurv) - heurv.sum()) / len(heurv) * 100, 
                        columns = ["perc_violations"]
                    ).T
                ])
                hname = heur if rule == "None" else f"{heur}-{rule}"
                heurv["method"] = hname
                heurv.reset_index(names = ["metric"], inplace = True)
                n_violations[wnb] = pd.concat(
                    [n_violations[wnb], heurv], ignore_index = True
                )
    return full_report, workload_boundaries, n_violations


def evaluate_scenario(
        scenario_dir: str, 
        heuristics: list, 
        target: str,
        skip: dict
    ) -> Tuple[dict, dict]:
    # create directory to store figures
    plot_dir = os.path.join(scenario_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all instances
    all_n_violations = {}
    for instance in os.listdir(scenario_dir):
        if instance.startswith("Instance"):
            print(f"    Processing results of {instance}")
            instance_dir = os.path.join(scenario_dir, instance)
            _, _, n_violations = evaluate_instance(
                instance_dir, 
                heuristics, 
                target,
                skip
            )
            all_n_violations[instance] = n_violations
    print("    **** Evaluating cumulative results")
    # count violations per instance
    metrics = {}
    for instance, instance_data in all_n_violations.items():
        for wnb, wnb_data in instance_data.items():
            if wnb not in metrics:
                metrics[wnb] = {}
            df = wnb_data[wnb_data["metric"] == "perc_violations"].set_index(
                "method", drop = True
            ).drop("metric", axis = "columns")
            for colname in df.columns:
                newcol = pd.DataFrame(df[colname]).rename(
                    columns = {colname: instance}
                )
                if colname not in metrics[wnb]:
                    metrics[wnb][colname] = newcol
                else:
                    metrics[wnb][colname] = metrics[wnb][colname].join(newcol)
    if not skip["scenario"]:
        for wnb, wnb_data in metrics.items():
            for metric, metric_data in wnb_data.items():
                metric_data = metric_data.sort_index(axis = "columns").T
                # plot percentage number of violations per instance
                barplot(
                    data=metric_data, 
                    ylabel="percentage # violations",
                    title=f"num_violations_extendedtrace-{metric}-w_{wnb}.png",
                    plot_dir=plot_dir
                )
    return all_n_violations, metrics


def evaluate_workload(
        lb_dir: str, 
        heuristics: list, 
        target: str,
        skip: dict
    ):
    # create directory to store figures
    plot_dir = os.path.join(lb_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all scenarios
    all_results = {}
    for scenario in os.listdir(lb_dir):
        if scenario.startswith("Scenario"):
            print(f"  Processing results of {scenario}")
            scenario_dir = os.path.join(lb_dir, scenario)
            _, metrics = evaluate_scenario(
                scenario_dir, 
                heuristics, 
                target, 
                skip
            )
            all_results[scenario] = metrics
            print("  ", "-"*77)
    print("  **** Evaluating cumulative results")
    # plot the average number of violations in all scenarios
    avg_n_violations = {}
    for scenario, scenario_data in all_results.items():
        for wnb, wnb_data in metrics.items():
            if wnb not in avg_n_violations:
                avg_n_violations[wnb] = {}
            for metric, metric_data in wnb_data.items():
                newcol = pd.DataFrame(
                    metric_data.mean(axis = "columns"), columns = [scenario]
                )
                if metric not in avg_n_violations[wnb]:
                    avg_n_violations[wnb][metric] = newcol
                else:
                    avg_n_violations[wnb][metric] = avg_n_violations[wnb][
                        metric
                    ].join(newcol)
    if not skip["workload"]:
        for wnb, wnb_data in avg_n_violations.items():
            for metric, metric_data in wnb_data.items():
                metric_data = metric_data.sort_index(axis = "columns").T
                barplot(
                    metric_data,
                    "average percentage # violations",
                    f"avg_n_violations_extendedtrace-{metric}-w_{wnb}.png",
                    plot_dir,
                    rot = 0
                )
    return all_results, avg_n_violations


def main(
        results_dir: str, 
        heuristics: list, 
        target: str,
        skip: dict
    ):
    # create directory to store figures
    plot_dir = os.path.join(results_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all max workload values
    all_n_violations = {}
    for lb_max_str in os.listdir(results_dir):
        if lb_max_str.startswith("Lambda_"):
            tokens = lb_max_str.split("-")
            lambda_max = float(tokens[0].replace("Lambda_", ""))
            bandwidth_min = float(tokens[1].replace("Bandwidth_", ""))
            print(
              f"Processing results with max workload {lambda_max} "
              f"and minimum bandwidth {bandwidth_min}"
            )
            lb_dir = os.path.join(results_dir, lb_max_str)
            _, n_violations = evaluate_workload(
                lb_dir, 
                heuristics, 
                target, 
                skip
            )
            # violations
            all_n_violations[lb_max_str] = n_violations
            print("#"*80)
    print("**** Evaluating cumulative results")
    # count the average number of violations
    avg_n_violations = {}
    for lb, lb_data in all_n_violations.items():
        for wnb, wnb_data in lb_data.items():
            if wnb not in avg_n_violations:
                avg_n_violations[wnb] = {}
            for metric, metric_data in wnb_data.items():
                newcol = pd.DataFrame(
                    metric_data.mean(axis = "columns"), columns = [lb]
                )
                if metric not in avg_n_violations[wnb]:
                    avg_n_violations[wnb][metric] = newcol
                else:
                    avg_n_violations[wnb][metric] = avg_n_violations[wnb][
                        metric
                    ].join(newcol)
    # plot
    for wnb, wnb_data in avg_n_violations.items():
        for metric, metric_data in wnb_data.items():
            metric_data.columns = [
                float(
                    c.split("-")[0].split("_")[1]
                ) for c in metric_data.columns
            ]
            metric_data = metric_data.sort_index(axis = "columns").T
            barplot(
                metric_data,
                "average percentage # violations",
                f"avg_n_violations_extendedtrace-{metric}-w_{wnb}.png",
                plot_dir,
                rot = 0,
                xlabel = "$\lambda_{\max} [req/s]$"
            )


if __name__ == "__main__":
    # parse arguments
    args = parse_arguments()
    results_dir = args.results_dir
    heuristics = args.heuristics
    target = args.target
    skip = {
        "instance": args.skip_instance, 
        "scenario": args.skip_scenario,
        "workload": args.skip_workload
    }
    # run
    main(results_dir, heuristics, target, skip)
