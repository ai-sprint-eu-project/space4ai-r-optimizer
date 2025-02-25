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

from external import space4ai_logger

from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Tuple
import networkx as nx
import numpy as np
import argparse
import json
import sys
import os


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
      description="Testing Scenario Generation"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--seed", 
      help="Seed for random number generation", 
      type=int,
      default=4850
    )
    parser.add_argument(
      "-v", "--verbose", 
      help="Verbosity level", 
      type=int,
      default=0
    )
    args, _ = parser.parse_known_args()
    return args


def read_configuration_file(config_file: str) -> dict:
    config = {}
    with open(config_file, "r") as istream:
        config = json.load(istream)
    return config


def check_configuration(config: dict) -> Tuple[int, str]:
    err_code = 0
    err = ""
    # get parameters
    n = config["n_components"]
    noe = config.get("n_only_edge_components", 0)
    noc = config.get("n_only_cloud_components", 0)
    # the number of local constraints must not exceed the number of components
    if config.get("max_n_local_constraints", 0) > config["n_components"]:
        err_code = 1
        err = "max_n_local_constraints > n_components"
    # the number of only-edge and only-cloud components should be compatible
    # with the number of components
    elif noe > n or noc > n or noe + noc > n:
        err_code = 2
        err = "n_only_edge, n_only_cloud or sum > n_components"
    # edge resources must exist if there are only-edge components
    elif noe > 0 and "edge_resources" not in config:
        err_code = 3
        err = "n_only_edge_components > 0 but no edge resources"
    # cloud resources must exist if there are only-cloud components
    elif noc > 0 and "cloud_resources" not in config:
        err_code = 3
        err = "n_only_cloud_components > 0 but no cloud resources"
    return err_code, err


def plot_application_graph(DAG: nx.DiGraph, plot_dir: str):
    # weight edges according to the attached probability
    elarge = [
      (u, v) for (u, v, d) in DAG.edges(data=True) if d["weight"] >= 0.5
    ]
    esmall = [
      (u, v) for (u, v, d) in DAG.edges(data=True) if d["weight"] < 0.5
    ]
    # define position of nodes
    pos = nx.shell_layout(DAG)
    # draw nodes
    if "start" in list(DAG) and "end" in list(DAG):
        node_list = list(DAG)
        node_color = ["k" if x=="start" or x=="end" else "gold" \
                      for x in node_list]
    else:
        node_color = "gold"
    nx.draw_networkx_nodes(DAG, pos, node_size=500, node_color=node_color)
    # edges
    nx.draw_networkx_edges(DAG, pos, edgelist=elarge, width=2, edge_color="k")
    nx.draw_networkx_edges(DAG, pos, edgelist=esmall, width=1, edge_color="k")
    # add node labels
    nx.draw_networkx_labels(DAG, pos, font_size=10, font_family="sans-serif")
    # add edge labels (weight)
    e_labels = {
        (u, v): f"{d['weight']:.2f}" for (u, v, d) in DAG.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
      DAG, pos, edge_labels=e_labels, font_size=10, font_family="sans-serif"
    )
    # other graph properties
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, "DAG.png"), 
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
    )


def generate_application_graph(
      n_components: int, 
      seed: int,
      linear_pipeline: bool = False
    ) -> Tuple[dict, nx.DiGraph]:
    if not linear_pipeline:
      # generate random DAG object (edges have random weights)
      p = 0.8 if n_components > 4 else 1.0
      done = False
      while not done:
          G = nx.gnp_random_graph(n_components-2, p, seed=seed, directed=True)
          if len(G.nodes()) > 1:
              DAG = nx.DiGraph(
                [(u,v,{"weight": np.random.random()}) for (u,v) in G.edges() if u<v]
              )
              done = nx.is_directed_acyclic_graph(DAG)
          else:
              DAG = nx.DiGraph(G)
              done = True
          seed += 1000
      # define node mapping (mapping = {0: "a", 1: "b", 2: "c"})
      mapping = {}
      for node in DAG:
          mapping[node] = int(node) + 1
      # rename nodes
      DAG = nx.relabel_nodes(DAG, mapping)
      # add source
      sources = [edge[0] for edge in DAG.in_degree if edge[1]==0]
      DAG.add_node(0)
      for node in sources:
          DAG.add_edge(0, node, weight=np.random.random())
      # add destination
      destinations = [edge[0] for edge in DAG.out_degree if edge[1]==0]
      des = len(DAG.nodes)
      DAG.add_node(des)
      for node in destinations:
          DAG.add_edge(node, des, weight=np.random.random())
    else:
      # generate "linear" DAG
      DAG = nx.DiGraph()
      DAG.add_node(0)
      for idx in range(1, n_components):
        DAG.add_node(idx)
        DAG.add_edge(idx - 1, idx, weight = 1)
    # read edge weights
    edge_list = list(DAG.in_edges())
    edge_weight_list = []
    for edge in edge_list:
        weight = DAG.get_edge_data(edge[0], edge[1])["weight"]
        edge_weight_list.append((edge[0], edge[1], weight))
    d = defaultdict(list)
    for k, *v in edge_weight_list:
        d[k].append(v)
    flat_dag = list(d.items())
    # generate dictionary of successors and transition probabilities
    nodes_dict = {}
    nodes = list(DAG.nodes)
    for node in nodes:
        nexts = [edges[1] for edges in flat_dag if edges[0]==node]
        node_data = {}
        next_list = []
        trans_list = []
        if len(nexts) > 0:
            rand_list = np.random.random(len(nexts[0]))
            Sum = sum(rand_list)
            idx = 0
            for des in nexts[0]:
                DAG[node][des[0]]["weight"] = rand_list[idx] / Sum
                next_list.append("c"+str(des[0]+1))
                trans_list.append(DAG[node][des[0]]["weight"])
                idx += 1
        node_data["next"] = next_list
        node_data["transition_probability"] = trans_list
        nodes_dict["c"+str(node+1)] = node_data
    # return
    return nodes_dict, DAG


def extract_random_int(min_value: int, max_value: int) -> int:
    value = min_value
    if max_value > min_value:
        value = int(np.random.randint(min_value, max_value))
    return value


def generate_partition(
      component_name: int,
      partition_idx: int,
      DAG_dict: dict,
      min_data_size: int,
      max_data_size: int,
      is_last_component: bool,
      is_last_partition: bool
    ) -> dict:
    partition = {}
    # the successor of the last partition is the next component
    if is_last_partition:
        # the last component does not have successors
        if is_last_component:
            partition["next"] = []
            partition["data_size"] = [
              extract_random_int(min_data_size,max_data_size)
            ]
        else:
            # get successors from DAG
            partition["next"] = DAG_dict[component_name]["next"]
            # generate data size
            data_size_list = []
            for _ in DAG_dict[component_name]["next"]:
                data_size_list.append(
                  extract_random_int(min_data_size,max_data_size)
                )
            partition["data_size"] = data_size_list
        # the early-exit probability of the last partition is 0
        partition["early_exit_probability"] = 0
    else:
        partition["next"] = [f"h{partition_idx+1}"]
        partition["data_size"] = [
          extract_random_int(min_data_size,max_data_size)
        ]
        partition["early_exit_probability"] = float(np.random.uniform(0,1))
    return partition


def generate_components(
      n_components: int, 
      max_n_deployments: int, 
      max_n_partitions: int,
      DAG_dict: dict,
      min_data_size: int,
      max_data_size: int
    ) -> dict:
    components = {}
    # loop over all components
    for i in range(1, n_components+1):
        component_name = f"c{i}"
        is_last_component = (i == n_components)
        # generate the first partition (whole component) and deployment
        first_partition = generate_partition(
          component_name=component_name, 
          partition_idx=1,
          DAG_dict=DAG_dict, 
          min_data_size=min_data_size, 
          max_data_size=max_data_size, 
          is_last_component=is_last_component,
          is_last_partition=True
        )
        deployments = {
          "s1": {"h1": first_partition}
        }
        # randomly choose the number of deployments
        next_partition_idx = 2
        if max_n_deployments > 1:
            n_deployments = extract_random_int(1, max_n_deployments)
            # randomly choose the number of partitions per deployment
            all_n_partitions = [
                extract_random_int(
                    1, max_n_partitions
                ) for _ in range(n_deployments)
            ]
            # loop over deployments
            for s in range(2, n_deployments+1):
                n_partitions = all_n_partitions.pop()
                # loop over all partitions
                partitions = {}
                for h in range(1, n_partitions+1):
                    is_last_partition = (h == n_partitions)
                    partitions[f"h{next_partition_idx}"] = generate_partition(
                        component_name=component_name,
                        partition_idx=h,
                        DAG_dict=DAG_dict,
                        min_data_size=min_data_size,
                        max_data_size=max_data_size,
                        is_last_component=is_last_component,
                        is_last_partition=is_last_partition
                    )
                    next_partition_idx += 1
                deployments[f"s{s}"] = partitions
        components[component_name] = deployments
    return components


def generate_edge_cloud_resources(
      basename: str,
      resources_config: dict, 
      n_components: int,
      first_layer_idx: int
    ) -> dict:
    # get configuration parameters
    n_layers = resources_config["n_layers"]
    max_n_resources = resources_config["max_n_resources"]
    min_n_instances = resources_config["min_n_instances"]
    max_n_instances = resources_config["max_n_instances"]
    min_memory = resources_config["memory"]
    max_memory = resources_config["memory"] * n_components
    min_cost = resources_config["min_cost"]
    max_cost = resources_config["max_cost"]
    max_n_cores = resources_config["max_n_cores"]
    # randomly choose the number of resources per layer
    all_n_resources = [
        extract_random_int(1, max_n_resources) for _ in range(n_layers)
    ]
    # loop over all layers
    resources = {}
    first_resource_idx = 1
    for l in range(first_layer_idx, n_layers+first_layer_idx):
        n_resources = all_n_resources.pop()
        # loop over all resources
        layer = {}
        for j in range(first_resource_idx, n_resources+first_resource_idx):
            resource = {
              "number": extract_random_int(min_n_instances, max_n_instances),
              "cost": float(np.random.uniform(min_cost, max_cost)),
              "memory": extract_random_int(min_memory, max_memory),
              "n_cores": extract_random_int(1, max_n_cores)
            }
            layer[f"{basename}{j}"] = resource
        resources[f"computationalLayer{l}"] = layer
        first_resource_idx += n_resources
    return resources


def generate_faas_resources(
      components: dict, 
      resources_config: dict,
      first_layer_idx: int
    ) -> Tuple[dict, list]:
    # get configuration parameters
    n_faas_per_component = resources_config["n_faas_per_component"]
    min_memory = resources_config["memory"]
    max_memory = resources_config["memory"] * len(components)
    min_cost = resources_config["min_cost"]
    max_cost = resources_config["max_cost"]
    # initialize (unique) layer
    layer = f"computationalLayer{first_layer_idx}"
    faas_resources = {layer: {"transition_cost": 0}}
    # loop over all components in the compatibility list
    first_idx = 1
    faas_per_component = []
    for i, n_faas in n_faas_per_component:
        faas = []
        # loop over all deployments
        deployments = components[f"c{i}"]
        for partitions in deployments.values():
            # loop over all partitions
            for _ in partitions:
                # generate the required number of candidate FaaS resources
                for j in range(first_idx, n_faas+first_idx):
                    resource = {
                      "cost": float(np.random.uniform(min_cost, max_cost)),
                      "memory": extract_random_int(min_memory, max_memory),
                      "idle_time_before_kill": 600
                    }
                    resource_name = f"F{j}"
                    faas_resources[layer][resource_name] = resource
                    faas.append(resource_name)
                first_idx += n_faas
        faas_per_component.append((i, faas))
    return faas_resources, faas_per_component


def generate_resources(system: dict, config: dict) -> Tuple[dict, list, list]:
    n_components = config["n_components"]
    all_layers = []
    # generate Edge resources
    n_edge_layers = 0
    if "edge_resources" in config:
        resources_config = config["edge_resources"]
        system["EdgeResources"] = generate_edge_cloud_resources(
          basename="EN",
          resources_config=resources_config,
          n_components=n_components,
          first_layer_idx=1
        )
        n_edge_layers = len(system["EdgeResources"])
        all_layers += list(system["EdgeResources"].keys())
    # generate Cloud resources
    n_cloud_layers = 0
    if "cloud_resources" in config:
        resources_config = config["cloud_resources"]
        system["CloudResources"] = generate_edge_cloud_resources(
          basename="VM",
          resources_config=resources_config,
          n_components=n_components,
          first_layer_idx=n_edge_layers+1
        )
        n_cloud_layers = len(system["CloudResources"])
        all_layers += list(system["CloudResources"].keys())
    # generate FaaS resources
    faas_per_component = []
    if "faas_resources" in config:
        resources_config = config["faas_resources"]
        system["FaaSResources"], faas_per_component = generate_faas_resources(
          components=system["Components"],
          resources_config=resources_config,
          first_layer_idx=n_edge_layers+n_cloud_layers+1
        )
        all_layers += list(system["FaaSResources"].keys())
    return system, faas_per_component, all_layers


def generate_component_matrices(
      component: str,
      components: list,
      max_n_candidates: int,
      resources_list: list,
      resources_data: dict,
      allow_colocation: bool,
      logger: space4ai_logger.Logger
    ) -> Tuple[dict, dict, set]:
    component_candidates = {}
    component_performance = {}
    used_resources = set()
    used_resources_indices = set()
    # loop over all deployments
    deployments = components[component]
    for _, partitions in deployments.items():
        # loop over all partitions
        for partition in partitions:
            # define the resources sub-list with the required number of 
            # candidates
            resources_sublist = []
            if max_n_candidates >= len(resources_list):
                if allow_colocation:
                    resources_sublist = resources_list
                else:
                    resources_sublist = filter_used_resources(
                      resources_list,
                      used_resources
                    )
            else:
                if allow_colocation:
                    avail = list(range(len(resources_list)))
                else:
                    avail = filter_used_resources(
                      list(range(len(resources_list))),
                      used_resources_indices
                    )
                for _ in range(max_n_candidates):
                    j = np.random.choice(avail)
                    avail.remove(j)
                    resources_sublist.append(resources_list[j])
            # loop over resources
            partition_candidates = []
            partition_performance = {}
            if len(resources_sublist) == 0:
              logger.err(
                "No candidate resources available! Make sure that at least "
                f"{max_n_candidates} * {len(components)-1} + 1 resources are "
                "defined if no colocation is allowed"
              )
            for resource in resources_sublist:
                # compatibility matrix entry
                entry = {
                  "resource": resource,
                  "memory": int(np.random.choice(
                    resources_data[resource]["memory"]
                  ))
                }
                partition_candidates.append(entry)
                # performance dictionary entry
                d = np.random.uniform(
                  resources_data[resource]["demand"]["min"], 
                  resources_data[resource]["demand"]["max"]
                )
                entry = {
                  "model": resources_data[resource]["model"],
                  "demand": float(d),
                  "meanTime": float(d)
                }
                partition_performance[resource] = entry
                used_resources.add(resource)
                used_resources_indices.add(resources_list.index(resource))
            # update component data
            component_candidates[partition] = partition_candidates
            component_performance[partition] = partition_performance
    return component_candidates, component_performance, used_resources


def get_resources_info(
      system: dict, 
      resource_type: str,
      memory_values: list,
      demand_range: list,
      model_name: str
    ) -> Tuple[list, dict]:
    resources_list = []
    resources_data = {}
    if resource_type in system:
        resources = system[resource_type]
        for l, layer_data in resources.items():
            for k in layer_data:
                resources_list.append(k)
                resources_data[k] = {
                  "model": model_name,
                  "memory": memory_values,
                  "demand": {
                    "min": demand_range[0],
                    "max": demand_range[1]
                  }
                }
    return resources_list, resources_data


def filter_used_resources(
      resources_list: list, 
      all_used_resources: set
    ) -> list:
    filtered_list = []
    for res in resources_list:
        if res not in all_used_resources:
            filtered_list.append(res)
    return filtered_list


def generate_matrices(
      system: dict, 
      config: dict,
      faas_per_component: list,
      logger: space4ai_logger.Logger
    ) -> Tuple[dict, dict]:
    components = system["Components"]
    # get configuration parameters
    n_only_edge_components = config.get("n_only_edge_components", 0)
    n_only_cloud_components = config.get("n_only_cloud_components", 0)
    n_only_faas_components = config.get("n_only_faas_components", 0)
    max_n_candidates = config["max_n_candidates"]
    allow_colocation = config["allow_colocation"]
    edge_demand_range = config.get("edge_demand_range", [0., 0.])
    cloud_demand_range = config.get("cloud_demand_range", [0., 0.])
    faas_warm_demand_range = config.get("faas_warm_demand_range", [0., 0.])
    faas_cold_demand_increase = config.get("faas_cold_demand_increase", 0.1)
    edge_memory_values = config.get("edge_memory_values", [0.])
    cloud_memory_values = config.get("cloud_memory_values", [0.])
    faas_memory_values = config.get("faas_memory_values", [0.])
    # list of Edge nodes and corresponding data
    edge_list, edge_data = get_resources_info(
      system=system, 
      resource_type="EdgeResources",
      memory_values=edge_memory_values, 
      demand_range=edge_demand_range,
      model_name="QTedge"
    )
    # list of Cloud VMs
    cloud_list, cloud_data = get_resources_info(
      system=system, 
      resource_type="CloudResources",
      memory_values=cloud_memory_values, 
      demand_range=cloud_demand_range,
      model_name="QTcloud"
    )
    # define matrices
    compatibility_matrix = {}
    demand_matrix = {}
    all_used_resources = set()
    # loop over components that consider only Edge resources (if any)
    next_component = 1
    for i in range(next_component, n_only_edge_components+next_component):
        component = f"c{i}"
        # filter used resources if co-location is not allowed
        if not allow_colocation:
            edge_list = filter_used_resources(edge_list, all_used_resources)
        # generate matrix entries
        candidates, performance, used_resources = generate_component_matrices(
          component=component,
          components=components,
          max_n_candidates=max_n_candidates,
          resources_list=edge_list,
          resources_data=edge_data,
          allow_colocation=allow_colocation,
          logger=logger
        )
        # update matrices
        compatibility_matrix[component] = candidates
        demand_matrix[component] = performance
        all_used_resources = all_used_resources.union(used_resources)
    next_component += n_only_edge_components
    # loop over components that consider both Edge and Cloud resources (if any)
    edge_cloud_list = edge_list + cloud_list
    edge_cloud_data = {**edge_data, **cloud_data}
    n_edge_cloud_components = len(components) - \
                                n_only_edge_components - \
                                  n_only_cloud_components - \
                                    n_only_faas_components
    for i in range(next_component, n_edge_cloud_components+next_component):
        component = f"c{i}"
        # filter used resources if co-location is not allowed
        if not allow_colocation:
            edge_cloud_list = filter_used_resources(
              edge_cloud_list, all_used_resources
            )
        # generate matrix entries
        candidates, performance, used_resources = generate_component_matrices(
          component=component,
          components=components,
          max_n_candidates=max_n_candidates,
          resources_list=edge_cloud_list,
          resources_data=edge_cloud_data,
          allow_colocation=allow_colocation
        )
        # update matrices
        compatibility_matrix[component] = candidates
        demand_matrix[component] = performance
        all_used_resources = all_used_resources.union(used_resources)
    next_component += n_edge_cloud_components
    # loop over components that consider only Cloud resources (if any)
    for i in range(next_component, n_only_cloud_components+next_component):
        component = f"c{i}"
        # filter used resources if co-location is not allowed
        if not allow_colocation:
            cloud_list = filter_used_resources(cloud_list, all_used_resources)
        # generate matrix entries
        candidates, performance, used_resources = generate_component_matrices(
          component=component,
          components=components,
          max_n_candidates=max_n_candidates,
          resources_list=cloud_list,
          resources_data=cloud_data,
          allow_colocation=allow_colocation
        )
        # update matrices
        compatibility_matrix[component] = candidates
        demand_matrix[component] = performance
        all_used_resources = all_used_resources.union(used_resources)
    next_component += n_only_cloud_components
    # add FaaS compatibility and performance info
    for i, compatible_faas in faas_per_component:
        component = f"c{i}"
        # compute the number of FaaS candidates to be assigned to each 
        # partition
        partitions = [
          h for h_list in components[component].values() for h in h_list
        ]
        n_faas_per_partition = int(len(compatible_faas) / len(partitions))
        # add component to compatibility and demand matrix, if not present
        if component not in compatibility_matrix:
            compatibility_matrix[component] = {}
            demand_matrix[component] = {}
        # loop over all partitions
        last_j = 0
        for partition in partitions:
            # add partition to compatibility and demand matrix, if not present
            if partition not in compatibility_matrix[component]:
                compatibility_matrix[component][partition] = []
            if partition not in demand_matrix[component]:
                demand_matrix[component][partition] = {}
            # loop over candidate FaaS resources
            for j in range(last_j, n_faas_per_partition+last_j):
                resource = compatible_faas[j]
                compatibility_matrix[component][partition].append({
                  "resource": resource,
                  "memory": int(np.random.choice(faas_memory_values))
                })
                dw = np.random.uniform(
                  faas_warm_demand_range[0], faas_warm_demand_range[1]
                )
                demand_matrix[component][partition][resource] = {
                  "model": "PACSLTK",
                  "demandWarm": float(dw),
                  "demandCold": float(dw * (1 + faas_cold_demand_increase))
                }
            last_j += n_faas_per_partition
    # return
    return compatibility_matrix, demand_matrix


def generate_local_constraints(
      DAG: nx.DiGraph, 
      max_n_local_constraints: int, 
      local_threshold_range: list,
      all_demands: dict
    ) -> dict:
    local_constraints = {}
    if max_n_local_constraints > 0:
        constraints = []
        components = []
        nodes = list(DAG.nodes)
        while len(constraints) < max_n_local_constraints:
            node = np.random.choice(nodes)
            if node not in components:
                components.append(node)
                threshold = np.random.uniform(
                  local_threshold_range[0], local_threshold_range[1]
                )
                constraints.append((node, threshold))
        for i, threshold in constraints:
            c = f"c{i+1}"
            avg_demand = sum(all_demands[c]) / len(all_demands[c])
            local_constraints[c] = {
              "local_res_time": float(threshold * avg_demand)
            }
    return local_constraints


def generate_global_constraint(source, dest, DAG):
    if source != dest:
        paths = [
          path for path in nx.all_simple_paths(DAG, source, dest)
        ]
        if len(paths) > 0:
            if len(paths) > 1:
                idx = np.random.randint(0, len(paths)-1)
            else:
                idx = 0
            return paths[idx]
    return None


def generate_global_constraints(
      DAG: nx.DiGraph, 
      max_n_global_constraints: int, 
      global_threshold_range: list,
      all_demands: dict,
      constrain_whole_application: bool
    ) -> dict:
    global_constraints = {}
    if max_n_global_constraints > 0:
        constraints = []
        GCs = []
        nodes = list(DAG.nodes)
        # add a constraint on the whole application, if required
        if constrain_whole_application:
            source = nodes[0]
            dest = nodes[-1]
            path = generate_global_constraint(source, dest, DAG)
            if path is not None:
                GCs.append(path)
                threshold = np.random.uniform(
                  global_threshold_range[0], global_threshold_range[1]
                )
                constraints.append((path, threshold))
                max_n_global_constraints -= 1
        # add other random constraints
        while len(constraints) < max_n_global_constraints:
            source = np.random.choice(nodes)
            dest = np.random.choice(nodes)
            path = generate_global_constraint(source, dest, DAG)
            if path is not None and path not in GCs:
                GCs.append(path)
                threshold = np.random.uniform(
                  global_threshold_range[0], global_threshold_range[1]
                )
                constraints.append((path, threshold))
        # create dictionary
        idx = 1
        for path, threshold in constraints:
            components = []
            avg_demand = 0.
            for i in path:
                c = f"c{i+1}"
                components.append(c)
                avg_demand += sum(all_demands[c]) / len(all_demands[c])
            global_constraints[f"p{idx}"] = {
              "components": components,
              "global_res_time": float(threshold * avg_demand)
            }
            idx += 1
    return global_constraints


def generate_network_domains(
      network_technology: list,
      all_layers: list
    ) -> dict:
    network_domains = {}
    idx = 1
    for layers_idx, access_delay, bandwidth in network_technology:
        layers = all_layers if layers_idx == "all" else [
          f"computationalLayer{l}" for l in layers_idx
        ]
        network_domains[f"ND{idx}"] = {
          "computationalLayers": layers,
          "AccessDelay": access_delay,
          "Bandwidth": bandwidth
        }
        idx += 1
    return network_domains


def get_all_demands(system_performance: dict) -> dict:
    all_demands = {}
    for component, partitions in system_performance.items():
        all_demands[component] = []
        for resources in partitions.values():
            for model_data in resources.values():
                if "demand" in model_data:
                    all_demands[component].append(model_data["demand"])
    return all_demands


def generate_instance(
      config: dict, 
      instance_dir: str, 
      seed: int,
      logger: space4ai_logger.Logger
    ) -> dict:
    # initialize system
    system = {}
    # generate DAG
    logger.log("Generate DAG", 1)
    n_components = config["n_components"]
    system["DirectedAcyclicGraph"], DAG = generate_application_graph(
      n_components, seed, config.get("linear_pipeline", False)
    )
    plot_application_graph(DAG, instance_dir)
    logger.log("*done", 2)
    if len(list(nx.selfloop_edges(DAG))) > 0:
        logger.err("DAG has self-loops")
    else:
        # generate components
        logger.log("Generate components", 1)
        max_n_deployments = config["max_n_deployments"]
        max_n_partitions = config["max_n_partitions"]
        min_data_size, max_data_size = config["data_size_range"]
        system["Components"] = generate_components(
          n_components,
          max_n_deployments,
          max_n_partitions,
          system["DirectedAcyclicGraph"],
          min_data_size,
          max_data_size
        )
        logger.log("*done", 2)
        # generate resources
        logger.log("Generate resources", 1)
        system, faas_per_component, all_layers = generate_resources(system, config)
        logger.log("*done", 2)
        # generate compatibility and performance dictionaries
        logger.log("Generate compatibility and performance dictionaries", 1)
        system["CompatibilityMatrix"], system["Performance"] = generate_matrices(
          system=system,
          config=config,
          faas_per_component=faas_per_component,
          logger=logger
        )
        logger.log("*done", 2)
        # generate local constraints
        logger.log("Generate local constraints", 1)
        max_n_local_constraints = min(
          config.get("max_n_local_constraints", 0),
          n_components
        )
        local_threshold_range = config.get("local_threshold_range", [0., 0.])
        all_demands = get_all_demands(system["Performance"])
        system["LocalConstraints"] = generate_local_constraints(
          DAG=DAG,
          max_n_local_constraints=max_n_local_constraints,
          local_threshold_range=local_threshold_range,
          all_demands=all_demands
        )
        logger.log("*done", 2)
        # generate global constraints
        logger.log("Generate global constraints", 1)
        max_n_global_constraints = config.get("max_n_global_constraints", 0)
        global_threshold_range = config.get("global_threshold_range", [0., 0.])
        constrain_whole_application = config.get(
          "constrain_whole_application", False
        )
        system["GlobalConstraints"] = generate_global_constraints(
          DAG=DAG,
          max_n_global_constraints=max_n_global_constraints,
          global_threshold_range=global_threshold_range,
          all_demands=all_demands,
          constrain_whole_application=constrain_whole_application
        )
        logger.log("*done", 2)
        # generate network domains
        logger.log("Generate network domains", 1)
        network_technology = config["network_technology"]
        system["NetworkTechnology"] = generate_network_domains(
          network_technology=network_technology,
          all_layers=all_layers
        )
        logger.log("*done", 2)
        # read time and workload
        logger.log("Read time and workload", 1)
        system["Time"] = config["time"]
        system["Lambda"] = config["lambda"]
        logger.log("*done", 2)
        # define system file
        system_file = os.path.join(instance_dir, "SystemFile.json")
        with open(system_file, "w") as ostream:
            json.dump(system, ostream, indent=2)
    return system


def main(
      application_dir: str, 
      seed: int, 
      logger: space4ai_logger.Logger
    ):
    # read configuration file
    config_file = os.path.join(application_dir, "config.json")
    config = read_configuration_file(config_file)
    # check if configuration is valid
    err_code, err_msg = check_configuration(config)
    if err_code != 0:
        logger.err(f"Invalid configuration. Exiting with message: {err_msg}")
        sys.exit(err_code)
    # loop over the required instances
    for n in range(config["n_instances"]):
        # generate directory
        instance_dir = os.path.join(application_dir, f"Instance{n}")
        os.makedirs(instance_dir, exist_ok=True)
        logger.log(f"Generating instance {n}. Saving files in {instance_dir}")
        # generate instance
        generate_instance(config, instance_dir, seed, logger)


if __name__ == "__main__":
    # parse arguments
    args = parse_arguments()
    application_dir = args.application_dir
    seed = args.seed
    verbose = args.verbose
    # set seed for random number generation
    np.random.seed(seed)
    # run
    logger = space4ai_logger.Logger(name="GenerateScenario", verbose=verbose)
    main(application_dir, seed, logger)
