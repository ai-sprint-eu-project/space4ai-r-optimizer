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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import argparse
import json
import os

TIME_STEP = 450 / 60


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Results Postprocessing")
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


def get_workload_profile(dir: str) -> pd.DataFrame:
    lambdas = []
    workload_file = os.path.join(dir, "LambdaValues.json")
    with open(workload_file, "r") as istream:
        lambdas = json.load(istream)["LambdaVec"]
    workload_profile = pd.DataFrame({
      "time": [i * TIME_STEP for i in range(len(lambdas))],
      "workload": lambdas,
      "workload_str": [str(l) for l in lambdas],
      "solution_idx": list(range(len(lambdas)))
    })
    return workload_profile


def parse_output(
      solution_file: str,
      is_design_time: bool = False
    ) -> Tuple[bool, float, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    # load solution
    solution = {}
    with open(solution_file, "r") as istream:
        solution = json.load(istream)
    resources = pd.DataFrame()
    response_times = pd.DataFrame()
    assignments = pd.DataFrame()
    GC = {}
    cost = None
    # read workload
    workload = solution["Lambda"]
    # check feasibility
    feasible = solution["feasible"]
    if feasible or "uheur" in solution_file:
        # read resources and utilization (only at runtime)
        if not is_design_time:
            resources = pd.DataFrame(solution["Resources"]).transpose()
        # read components assignment and response times
        response_times = {}
        assignments = {
          "component": [], 
          "deployment": [], 
          "partition": [], 
          "resource": [],
          "response_time": []
        }
        for component, component_data in solution["components"].items():
            r = component_data.pop("response_time")
            response_times[component] = {
              "response_time": r if r is not None and r > 0.0 else None,
              "threshold": component_data.pop("response_time_threshold")
            }
            for deployment, deployment_data in component_data.items():
                h = 0
                for partition, assignment in deployment_data.items():
                    assignments["component"].append(component)
                    assignments["deployment"].append(deployment)
                    assignments["partition"].append(partition)
                    assignments["response_time"].append(assignment.pop(
                        "response_time"
                    ))
                    res = list(assignment.values())[0]
                    assignments["resource"].append(list(res.keys())[0])
                    # at design-time, read the number of selected instances
                    if is_design_time:
                        res_name = assignments["resource"][-1]
                        resource = pd.DataFrame({
                            "res": [res_name],
                            "cost": [res[res_name]["cost"]],
                            "description": [res[res_name]["description"]],
                            "memory": [res[res_name]["memory"]],
                            "number": [res[res_name]["number"]],
                            "utilization": [None],
                            "solution_idx": [0]
                        }).set_index("res", drop=True)
                        resources = pd.concat([resources, resource])
                    h += 1
        response_times = pd.DataFrame(response_times).transpose()
        assignments = pd.DataFrame(assignments)
        # read global constraints
        GC = solution["global_constraints"]
        # read cost
        cost = solution["total_cost"]
    return feasible, resources, response_times, assignments, GC, cost


def get_design_time_solution(solution_dir: str, workload: float):
    solution_file = os.path.join(solution_dir, f"Lambda_{workload}.json")
    if os.path.exists(solution_file):
        return parse_output(solution_file, True)
    return None


def get_solutions(solution_dir: str, workload_profile: pd.DataFrame):
    # loop over directory content
    resources = pd.DataFrame()
    response_times = {"local": pd.DataFrame(), "global": {}}
    assignments = pd.DataFrame()
    feasibility = []
    cost = []
    solution_idx = 1
    for workload in workload_profile.iloc[1:]["workload_str"]:
        filename = f"Lambda_{workload}.json"
        solution_file = os.path.join(solution_dir, filename)
        if os.path.exists(solution_file):
            feasible, r, rt, a, GC, c = parse_output(solution_file)
            r["solution_idx"] = [solution_idx] * len(r)
            rt["solution_idx"] = [solution_idx] * len(rt)
            a["solution_idx"] = [solution_idx] * len(a)
            feasibility.append(feasible)
            cost.append(c)
            resources = pd.concat([resources, r])
            response_times["local"] = pd.concat([response_times["local"], rt])
            response_times["global"][solution_idx] = GC
            assignments = pd.concat([assignments, a], ignore_index=True)
        else:
            feasibility.append(False)
            cost.append(None)
            temp = pd.DataFrame({"solution_idx": [solution_idx]})
            resources = pd.concat([resources, temp])
            response_times["local"] = pd.concat([response_times["local"], temp])
            response_times["global"][solution_idx] = None
            assignments = pd.concat([assignments, temp], ignore_index=True)
        solution_idx += 1
    return resources, response_times, assignments, cost, feasibility


def plot_workload_profile(
        workload_profile: pd.DataFrame, 
        lambda_max: float,
        plot_dir: str
    ):
    _, ax = plt.subplots()
    workload_profile.iloc[1:].plot(
      x = "time",
      y = "workload",
      marker = ".",
      fontsize = 14,
      ax = ax,
      linewidth = 2,
      label = "$\lambda$"
    )
    ax.axhline(
      y = lambda_max,
      linestyle = "dashed",
      color = mcolors.TABLEAU_COLORS["tab:red"],
      linewidth = 2,
      label = "$\lambda_{max}$"
    )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel("workload [req/s]", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, "workload.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def plot_utilization(
        results: pd.DataFrame, 
        workload_profile: pd.DataFrame, 
        heuristic: str,
        plot_dir: str,
        min_utilization: float = None,
        max_utilization: float = None
    ):
    column = "utilization"
    _, ax = plt.subplots()
    for key, data in results.groupby(results.index):
        if key != 0:
            df = pd.DataFrame({
                "time": list(workload_profile.iloc[1:]["time"]),
                "solution_idx": list(workload_profile.iloc[1:]["solution_idx"])
            }).join(
                pd.DataFrame({
                    "solution_idx": data["solution_idx"],
                    column: data[column]
                }).set_index("solution_idx"),
                on = "solution_idx"
            )
            df.plot(
                x = "time",
                y = column,
                ax = ax,
                label = key,
                marker = ".",
                fontsize = 14,
                grid = True
            )
            ax.axhline(
                y = 1,
                linestyle = "dashed",
                color = "red",
                linewidth = 2
            )
            if min_utilization is not None:
                ax.axhline(
                    y = min_utilization,
                    linestyle = "dotted",
                    color = mcolors.TABLEAU_COLORS["tab:red"],
                    linewidth = 1
                )
            if max_utilization is not None:
                ax.axhline(
                    y = max_utilization,
                    linestyle = "dotted",
                    color = mcolors.TABLEAU_COLORS["tab:red"],
                    linewidth = 1
                )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel(column, fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, f"utilization_{heuristic}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def plot_expected_utilization(
        results: pd.DataFrame, 
        workload_profile: pd.DataFrame, 
        heuristic: str,
        design_time_solution: pd.DataFrame,
        plot_dir: str,
        min_utilization: float = None,
        max_utilization: float = None
    ):
    column = "utilization"
    _, ax = plt.subplots()
    for key, data in results.groupby(results.index):
        if key != 0:
            n = [design_time_solution[1].loc[key]["number"]] + list(data.iloc[:-1]["number"])
            u = data["utilization"] * data["number"] / n
            df = pd.DataFrame({
                "time": list(workload_profile.iloc[1:]["time"]),
                "solution_idx": list(workload_profile.iloc[1:]["solution_idx"])
            }).join(
                pd.DataFrame({
                    "solution_idx": data["solution_idx"],
                    column: u
                }).set_index("solution_idx"),
                on = "solution_idx"
            )
            df.plot(
                x = "time",
                y = column,
                ax = ax,
                label = key,
                marker = ".",
                fontsize = 14,
                grid = True
            )
            ax.axhline(
                y = 1,
                linestyle = "dashed",
                color = "red",
                linewidth = 2
            )
            if min_utilization is not None:
                ax.axhline(
                    y = min_utilization,
                    linestyle = "dotted",
                    color = mcolors.TABLEAU_COLORS["tab:red"],
                    linewidth = 1
                )
            if max_utilization is not None:
                ax.axhline(
                    y = max_utilization,
                    linestyle = "dotted",
                    color = mcolors.TABLEAU_COLORS["tab:red"],
                    linewidth = 1
                )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel(column, fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, f"expected_utilization_{heuristic}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def plot_response_times(
      results: pd.DataFrame, 
      global_constraints: pd.DataFrame,
      workload_profile: pd.DataFrame, 
      heuristic: str,
      plot_dir: str
    ):
    column = "response_time"
    # plot response times by component
    _, ax = plt.subplots()
    for key, data in results.groupby(results.index):
        if key != 0:
            df = pd.DataFrame({
                "time": list(workload_profile.iloc[1:]["time"]),
                "solution_idx": list(workload_profile.iloc[1:]["solution_idx"])
            }).join(
                pd.DataFrame({
                    "solution_idx": data["solution_idx"],
                    column: data[column]
                }).set_index("solution_idx"),
                on = "solution_idx"
            )
            threshold = data["threshold"].unique()[0]
            df.plot(
                x = "time",
                y = column,
                ax = ax,
                label = key,
                marker = ".",
                fontsize = 14,
                grid = True
            )
            ax.axhline(
                y = threshold,
                linestyle = "dashed",
                color = plt.gca().lines[-1].get_color(),
                linewidth = "2"
            )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel("response time [s]", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, f"response_times_{heuristic}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()
    # plot total response times if global constraints are provided
    if len(global_constraints) > 0:
        _, ax = plt.subplots()
        for path in global_constraints.groupby(global_constraints.index):
            path_name = path[0]
            df = pd.DataFrame({
                "time": list(workload_profile.iloc[1:]["time"]),
                "solution_idx": list(workload_profile.iloc[1:]["solution_idx"])
            }).join(
                pd.DataFrame({
                    "solution_idx": path[1]["solution_idx"],
                    "path_response_time": path[1]["path_response_time"],
                    "components": path[1]["components"]
                }).set_index("solution_idx"),
                on = "solution_idx"
            )
            threshold = path[1]["global_res_time"].unique()[0]
            df.plot(
                x = "time",
                y = "path_response_time",
                label = f"{path_name}: {df['components'].iloc[0]}",
                marker = ".",
                linewidth = 0.1,
                fontsize = 14,
                ax = ax
            )
            ax.axhline(
                y = threshold,
                linestyle = "dashed",
                linewidth = 2,
                color = mcolors.TABLEAU_COLORS["tab:red"]
                #plt.gca().lines[-1].get_color() if more than one path
            )
        ax.set_xlabel("time [min]", fontsize = 14)
        ax.set_ylabel("response time [s]", fontsize = 14)
        ax.legend(fontsize = 14)
        plt.savefig(
            os.path.join(plot_dir, f"total_response_times_{heuristic}.png"),
            dpi = 300,
            format = "png",
            bbox_inches = "tight"
        )
        plt.close()


def plot_n_instances(
      results: pd.DataFrame, 
      workload_profile: pd.DataFrame, 
      heuristic: str,
      design_time_solution: pd.DataFrame,
      plot_dir: str,
    ):
    column = "number"
    _, ax = plt.subplots()
    for key, data in results.groupby(results.index):
        if key != 0:
            n_dt = design_time_solution[1].loc[key]["number"]
            df = pd.DataFrame({
                "time": list(workload_profile.iloc[1:]["time"]),
                "solution_idx": list(workload_profile.iloc[1:]["solution_idx"])
            }).join(
                pd.DataFrame({
                    "solution_idx": data["solution_idx"],
                    column: data[column]
                }).set_index("solution_idx"),
                on = "solution_idx"
            )
            df.plot(
                x = "time",
                y = column,
                ax = ax,
                label = key,
                marker = ".",
                fontsize = 14,
                grid = True
            )
            ax.plot(
                0,
                n_dt,
                "x",
                markersize = 10,
                color = plt.gca().lines[-1].get_color()
            )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel("number of instances", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, f"n_instances_{heuristic}.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def evaluate_heuristic(
        dir: str, 
        workload_profile: pd.DataFrame,
        heuristic: str,
        design_time_solution: pd.DataFrame,
        plot_dir: str,
        min_utilization: float,
        max_utilization: float,
        skip: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    # get solution data
    solution_dir = os.path.join(dir, heuristic)
    resources, response_times, assignments, cost, feasibility = get_solutions(
      solution_dir, workload_profile
    )
    # define global constraints
    global_constraints = pd.DataFrame()
    for solution_idx, GC in response_times["global"].items():
        df = pd.DataFrame(GC).transpose()
        df["solution_idx"] = [solution_idx] * len(df)
        global_constraints = pd.concat([global_constraints, df])
    if not skip["instance"]:
        # plot results
        plot_utilization(
            resources, 
            workload_profile, 
            heuristic, 
            plot_dir,
            min_utilization,
            max_utilization
        )
        plot_response_times(
            response_times["local"],
            global_constraints, 
            workload_profile, 
            heuristic, 
            plot_dir
        )
        plot_n_instances(
            resources, 
            workload_profile, 
            heuristic, 
            design_time_solution, 
            plot_dir
        )
        plot_expected_utilization(
            resources, 
            workload_profile, 
            heuristic,
            design_time_solution, 
            plot_dir,
            min_utilization,
            max_utilization
        )
    # return
    return resources, response_times["local"], assignments, cost, feasibility


def plot_costs(costs: dict, workload_profile: pd.DataFrame, plot_dir: str):
    _, ax = plt.subplots()
    for heuristic, cost in costs.items():
        df = pd.DataFrame({
            "time": workload_profile.iloc[1:]["time"],
            "cost": cost
        })
        df.plot(
            x = "time",
            y = "cost",
            label = heuristic,
            ax = ax,
            marker = ".",
            linewidth = 2,
            grid = True,
            fontsize = 14
        )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel("cost [$]", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, "costs.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def compute_percentage_cost_reduction(costs: dict, target: str) -> dict:
    pcr = {}
    for heuristic, cost in costs.items():
        if heuristic != target:
            pcr[heuristic] = []
            for t,o in zip(costs[target], cost):
                if t is not None and o is not None:
                    pcr[heuristic].append((o-t)/o * 100)
                else:
                    pcr[heuristic].append(None)
    return pcr


def plot_percentage_cost_reduction(
        pcr: dict, 
        workload_profile: pd.DataFrame,
        target: str,
        plot_dir: str
    ):
    _, ax = plt.subplots()
    for heuristic, pcr in pcr.items():
        df = pd.DataFrame({
            "time": workload_profile.iloc[1:]["time"],
            "pcr": pcr
        })
        df.plot(
            x = "time",
            y = "pcr",
            label = heuristic,
            ax = ax,
            marker = ".",
            linewidth = 2,
            grid = True,
            fontsize = 14
        )
    ax.set_xlabel("time [min]", fontsize = 14)
    ax.set_ylabel(f"percentage cost reduction of {target}", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, "pcr.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def barplot(data: pd.DataFrame, ylabel: str, title: str, plot_dir: str):
    ax = data.plot.bar(
        fontsize = 14
    )
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=14)
    plt.savefig(
        os.path.join(plot_dir, title),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def plot_average_costs_over_time(costs: pd.DataFrame, plot_dir: str):
    time = [i * TIME_STEP for i in range(1, len(costs)+1)]
    costs["time"] = time
    ax = costs.plot(
        x = "time",
        marker = ".",
        linewidth = 2,
        fontsize = 14
    )
    ax.set_xlabel("time [min]", fontsize=14)
    ax.set_ylabel("average costs [$]", fontsize=14)
    ax.legend(fontsize=14)
    plt.savefig(
        os.path.join(plot_dir, "average_costs_over_time.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def evaluate_instance(
        instance_dir: str, 
        heuristics: list, 
        target: str,
        min_utilization: float,
        max_utilization: float,
        skip: dict
    ) -> Tuple[dict, dict, dict]:
    # create directory to store figures
    plot_dir = os.path.join(instance_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # get and plot workload profile
    workload_profile = get_workload_profile(instance_dir)
    lambda_max = workload_profile.iloc[0]["workload_str"]
    plot_workload_profile(workload_profile, float(lambda_max), plot_dir)
    # get design-time solution
    design_time_solution = get_design_time_solution(instance_dir, lambda_max)
    if design_time_solution is not None:
        # get and plot heuristics result
        results = {}
        costs = {}
        for heuristic in heuristics:
            if heuristic == "figaro":
                results[heuristic] = None
                c = pd.read_csv(
                    os.path.join(
                        instance_dir, "figaro/processed_data/costs.csv"
                    )
                )
                costs[heuristic] = list(c["cost"])
            else:
                # results:
                # (resources, local_resp_times, assignments, cost, feasibility)
                results[heuristic] = evaluate_heuristic(
                    instance_dir, 
                    workload_profile, 
                    heuristic, 
                    design_time_solution, 
                    plot_dir,
                    min_utilization,
                    max_utilization,
                    skip
                )
                costs[heuristic] = results[heuristic][3]
        if not skip["instance"]:
            # plot costs
            plot_costs(costs, workload_profile, plot_dir)
        # compute and plot percentage cost reduction
        pcr = compute_percentage_cost_reduction(costs, target)
        if not skip["instance"]:
            plot_percentage_cost_reduction(
                pcr, workload_profile, target, plot_dir
            )
        return results, costs, pcr
    else:
        print("      !!!! WARNING: design-time solution not found !!!!")
    return None


def compute_avg(values: list):
    avg = 0.
    for value in values:
        if value is not None:
            avg += value
    return avg / len(values)


def dict_to_avgdf(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data).transpose()
    for column in df.columns:
        df[f"{column}_avg"] = [compute_avg(s) for s in df[column]]
    df_avg = pd.DataFrame(
        {k: v for (k,v) in df.items() if k.endswith("_avg")}
    )
    return df_avg


def evaluate_scenario(
        scenario_dir: str, 
        heuristics: list, 
        target: str,
        min_utilization: float,
        max_utilization: float,
        skip: dict
    ) -> Tuple[dict, dict, dict, pd.DataFrame]:
    # create directory to store figures
    plot_dir = os.path.join(scenario_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all instances
    all_results = {}
    all_costs = {}
    all_pcr = {}
    for instance in os.listdir(scenario_dir):
        if instance.startswith("Instance"):
            print(f"    Processing results of {instance}")
            instance_dir = os.path.join(scenario_dir, instance)
            rrr = evaluate_instance(
                instance_dir, 
                heuristics, 
                target,
                min_utilization, 
                max_utilization,
                skip
            )
            if rrr is not None:
                # results:
                # (resources, local_resp_times, assignments, cost, feasibility)
                results, costs, pcr = rrr
                all_results[instance] = results
                all_costs[instance] = costs
                all_pcr[instance] = pcr
    print("    **** Evaluating cumulative results")
    # compute and plot average percentage cost reduction in all instances
    print("      compute average percentage cost reduction in each instance")
    all_pcr = {
        "all_pcr": all_pcr,
        "all_pcr_avg": dict_to_avgdf(all_pcr)
    }
    if not skip["scenario"]:
        barplot(
            data=all_pcr["all_pcr_avg"], 
            ylabel="average percentage cost reduction", 
            title="average_pcr.png", 
            plot_dir=plot_dir
        )
    # compute and plot average costs for all instances
    print("      compute average cost in each instance")
    all_costs = {
        "all_costs": all_costs,
        "all_costs_avg": dict_to_avgdf(all_costs)
    }
    if not skip["scenario"]:
        barplot(
            data=all_costs["all_costs_avg"], 
            ylabel="average costs [$]",
            title="average_costs.png",
            plot_dir=plot_dir
        )
    # compute average cost across instances and number of violations per instance
    print("      compute average cost and #violations across instances")
    d = {heuristic: {} for heuristic in heuristics}
    n_violations = {}
    for heuristic in heuristics:
        f = {}
        for instance, instance_data in all_results.items():
            if heuristic != "figaro":
                f[instance] = instance_data[heuristic][-1]
            d[heuristic][instance]=all_costs["all_costs"][instance][heuristic]
        f = pd.DataFrame(f)
        n_violations[heuristic] = ((len(f) - f.sum()) / len(f)) * 100
        d[heuristic] = pd.DataFrame(d[heuristic]).mean(axis=1)
    all_costs["all_costs_avg_over_time"] = pd.DataFrame(d)
    n_violations = pd.DataFrame(n_violations)
    if not skip["scenario"]:
        # plot average cost across instances
        plot_average_costs_over_time(
            all_costs["all_costs_avg_over_time"], 
            plot_dir
        )
        # plot number of violations per instance
        barplot(
            data=n_violations, 
            ylabel="percentage # violations",
            title="num_violations.png",
            plot_dir=plot_dir
        )
    return all_results, all_costs, all_pcr, n_violations


def evaluate_workload(
        lambda_dir: str, 
        heuristics: list, 
        target: str,
        min_utilization,
        max_utilization,
        skip: dict
    ):
    # create directory to store figures
    plot_dir = os.path.join(lambda_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all scenarios
    all_results = {}
    all_costs = {}
    all_pcr = {}
    all_pcr_df = pd.DataFrame()
    all_n_violations = pd.DataFrame()
    for scenario in os.listdir(lambda_dir):
        if scenario.startswith("Scenario"):
            print(f"  Processing results of {scenario}")
            scenario_dir = os.path.join(lambda_dir, scenario)
            results, costs, pcr, n_violations = evaluate_scenario(
                scenario_dir, 
                heuristics, 
                target, 
                min_utilization, 
                max_utilization,
                skip
            )
            all_results[scenario] = results
            all_costs[scenario] = costs
            # percentage cost reduction
            all_pcr[scenario] = pcr
            df = pd.DataFrame(pcr["all_pcr_avg"])
            df["instances"] = df.index
            df["scenario"] = [scenario] * len(df)
            all_pcr_df = pd.concat(
                [all_pcr_df, df], ignore_index=True
            )
            # violations
            n_violations = pd.DataFrame(n_violations)
            n_violations["instances"] = n_violations.index
            n_violations["scenario"] = [scenario] * len(n_violations)
            all_n_violations = pd.concat(
                [all_n_violations, n_violations], ignore_index=True
            )
            print("  ", "-"*77)
    print("  **** Evaluating cumulative results")
    # plot the average percentage cost reduction in all scenarios
    heuristics_no_target = [h for h in heuristics if h != target]
    all_pcr = {
        "all_pcr": all_pcr,
        "all_pcr_avg": all_pcr_df.groupby("scenario")[
            [f"{h}_avg" for h in heuristics_no_target]
        ].mean()
    }
    if not skip["workload"]:
        barplot(
            all_pcr["all_pcr_avg"],
            "average percentage cost reduction",
            "average_pcr.png",
            plot_dir
        )
    # plot the average number of violations in all scenarios
    no_figaro_heur = [h for h in heuristics if h != "figaro"]
    all_n_violations = {
        "all_n_violations": all_n_violations,
        "all_n_violations_avg": all_n_violations.groupby(
            "scenario"
        )[no_figaro_heur].mean()
    }
    if not skip["workload"]:
        barplot(
            all_n_violations["all_n_violations_avg"],
            "average percentage # violations",
            "avg_n_violations.png",
            plot_dir
        )
    return all_results, all_costs, all_pcr, all_n_violations


def evaluate_rule(
        results_dir: str, 
        heuristics: list, 
        target: str,
        min_utilization: float,
        max_utilization: float,
        skip: dict
    ):
    # create directory to store figures
    plot_dir = os.path.join(results_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    # loop over all max workload values
    all_pcr = {}
    all_pcr_df = pd.DataFrame()
    all_n_violations = {}
    all_n_violations_df = pd.DataFrame()
    for lambda_max_str in os.listdir(results_dir):
        if lambda_max_str.startswith("Lambda_"):
            lambda_max = float(lambda_max_str.split("Lambda_")[1])
            print(f"Processing results with max workload {lambda_max}")
            lambda_dir = os.path.join(results_dir, lambda_max_str)
            _, _, pcr, n_violations = evaluate_workload(
                lambda_dir, 
                heuristics, 
                target, 
                min_utilization, 
                max_utilization,
                skip
            )
            # percentage cost reduction
            all_pcr[lambda_max_str] = pcr
            df = pd.DataFrame(pcr["all_pcr_avg"])
            df["scenario"] = df.index
            df["lambda_max"] = [lambda_max] * len(df)
            all_pcr_df = pd.concat(
                [all_pcr_df, df], ignore_index=True
            )
            # violations
            n_violations[lambda_max_str] = n_violations
            n_violations_df = pd.DataFrame(n_violations["all_n_violations_avg"])
            n_violations_df["scenario"] = n_violations_df.index
            n_violations_df["lambda_max"] = [lambda_max] * len(n_violations_df)
            all_n_violations_df = pd.concat(
                [all_n_violations_df, n_violations_df], ignore_index=True
            )
            print("#"*80)
    print("**** Evaluating cumulative results")
    # plot the average percentage cost reduction for all workloads
    heuristics_no_target = [h for h in heuristics if h != target]
    all_pcr = {
        "all_pcr": all_pcr,
        "all_pcr_avg": all_pcr_df.groupby("lambda_max")[
            [f"{h}_avg" for h in heuristics_no_target]
        ].mean()
    }
    barplot(
        all_pcr["all_pcr_avg"],
        "average percentage cost reduction",
        "average_pcr.png",
        plot_dir
    )
    # plot the average percentage cost reduction in all scenarios
    alldf = pd.DataFrame()
    for lambda_str in all_pcr["all_pcr"]:
        df = all_pcr["all_pcr"][lambda_str]["all_pcr_avg"].transpose()
        lambda_max = df.loc["lambda_max"].unique()[0]
        df = df.drop(["scenario", "lambda_max"])
        df["lambda_max"] = [lambda_max] * len(df)
        alldf = pd.concat([alldf, df])
    alldf = alldf.sort_values("lambda_max")
    ax = alldf.plot.bar(
        x = "lambda_max",
        rot = 0,
        fontsize = 14
    )
    ax.set_xlabel(None)
    ax.set_ylabel("average percentage cost reduction", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, "average_pcr_all_scenarios.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()
    # plot the average number of violations in all scenarios
    no_figaro_heur = [h for h in heuristics if h != "figaro"]
    all_n_violations = {
        "all_n_violations": all_n_violations_df,
        "all_n_violations_avg": all_n_violations_df.groupby(
            "lambda_max"
        )[no_figaro_heur].mean()
    }
    barplot(
        all_n_violations["all_n_violations_avg"],
        "average percentage # violations",
        "avg_n_violations.png",
        plot_dir
    )
    # plot the average number of violations in all scenarios
    alldf = pd.DataFrame()
    for lambda_val in all_n_violations["all_n_violations"].groupby("lambda_max"):
        lambda_max = lambda_val[0]
        df = lambda_val[1].set_index("scenario").transpose()
        df = df.drop(["lambda_max"])
        df["lambda_max"] = [lambda_max] * len(df)
        alldf = pd.concat([alldf, df])
    alldf = alldf.sort_values("lambda_max")
    alldf["test"] = [
        f"{alldf['lambda_max'].iloc[i]}\n{alldf.index[i]}" for i in range(len(alldf))
    ]
    alldf = alldf.drop("lambda_max", axis=1)
    ax = alldf.plot.bar(
        x = "test",
        rot = 0,
        fontsize = 14
    )
    ax.set_xlabel(None)
    ax.set_ylabel("average percentage # violations", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.savefig(
        os.path.join(plot_dir, "avg_n_violations_all_scenarios.png"),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )
    plt.close()


def main(
        all_results_dir: str, 
        heuristics: list, 
        target: str,
        skip: dict
    ):
    # loop over all subfolders
    for name in os.listdir(all_results_dir):
        if name.startswith("fixed_") or name.startswith("percentage_"):
            tokens = name.split("_")
            rule = tokens[0]
            min_utilization = float(tokens[1])
            max_utilization = float(tokens[2])
            if len(tokens) > 3:
                decr_percentage = float(tokens[3])
                incr_percentage = float(tokens[4])
            print(f"Processing folder {name}")
            results_dir = os.path.join(all_results_dir, name)
            # evaluate the current results
            evaluate_rule(
                results_dir, 
                heuristics, 
                target, 
                min_utilization, 
                max_utilization,
                skip
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
