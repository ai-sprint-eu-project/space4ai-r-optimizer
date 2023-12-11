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

from external import space4ai_logger, space4ai_parser

from flask import Flask, request, jsonify
from waitress import serve
from typing import Tuple
import shutil
import yaml
import json
import os

# ----------------------------------------------------------------------------
# global definitions
# ----------------------------------------------------------------------------
app = Flask(__name__)

# online path
home_path = "/"
path = "/space4air/workload"
json_path = "/space4air/workload/json"

# exit codes
NOT_FOUND = 404
SUCCESS = 201

# error messages
error_msg = {
    404: "ERROR: page not found",
    414: "ERROR: missing mandatory input `application_dir`",
    424: "ERROR: `application_dir` is not accessible",
    434: "ERROR: `production_deployment` is not accessible",
    444: "ERROR: missing mandatory input `lowerBoundLambda` in json setting",
    454: "ERROR: system file is not accessible",
    464: "ERROR: current solution file is not accessible",
    474: "ERROR: `upperBoundLambda` or `epsilon` not properly set"
}

# mount point
MOUNT_POINT = os.getenv("MOUNT_POINT", "/mnt")


# ----------------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------------
def get_binary_search_parameters(
      application_dir: str
    ) -> Tuple[float, float, float]:
    """
    Get the binary search parameter used at design-time
    """
    min_lambda = 0.0
    max_lambda = 0.0
    epsilon = 0.0
    space4aid_path = os.path.join(application_dir, "space4ai-d")
    # open system file to get the minimum workload
    design_time_system_file = os.path.join(space4aid_path, "SystemFile.json")
    with open(design_time_system_file, "r") as file:
        system = json.load(file)
        min_lambda = system["Lambda"]
    # open configuration file to get the maximum workload and epsilon
    design_time_config_file = os.path.join(space4aid_path, "SPACE4AI-D.yaml")
    BS_names_list = ["BS", "BinarySearch", "binary_search", "binarysearch"]
    with open(design_time_config_file, "r") as file:
        config = yaml.full_load(file)
        # get upper bound and epsilon
        for method_parameters in config["Methods"].values():
            if method_parameters["name"] in BS_names_list:
                max_lambda = method_parameters["upperBoundLambda"]
                epsilon = method_parameters["epsilon"]
    return min_lambda, max_lambda, epsilon


def change_format(application_dir: str, production_deployment: dict) -> dict:
    """
    Convert the production deployment into the format required by the 
    space4ai-r optimizer
    """
    # initialize parser
    parser = space4ai_parser.ParserYamlToJson(
        application_dir=application_dir, who="s4ai-r"
    )
    # load relevant information from the production deployment
    selected_components = parser.get_selected_components(production_deployment)
    # define solution
    current_solution = {
        "components": selected_components,
        "global_constraints": None,
        "total_cost": None
    }
    return current_solution


def coincide(existing_components: dict, new_components: dict) -> bool:
    """
    Return True if two solutions include the same selected deployments and 
    placements
    """
    # loop over all components
    for c, component_data in existing_components.items():
        # exit if the component is not in the second solution
        if c not in new_components:
            return False
        # loop over all deployments
        for s, deployment_data in component_data.items():
            # exit if the deployment is not in the second solution
            if s not in new_components[c]:
                return False
            # loop over all partitions
            for h, partition_data in deployment_data.items():
                # exit if the deployment is not in the second solution
                if h not in new_components[c][s]:
                    return False
                # get the selected computational layer (unique!)
                l = list(partition_data.keys())[0]
                # exit if the layer is not in the second solution
                if l not in new_components[c][s][h]:
                    return False
                # get the selected resource (unique!)
                r = list(partition_data[l].keys())[0]
                # exit if the resources do not coincide
                if r not in new_components[c][s][h][l]:
                    return False
    return True


def read_max_workload(application_dir: str, current_solution: dict) -> float:
    """
    Read the maximum workload from the files provided at design-time (if the 
    current solution is one of the considered ones)
    """
    highest_feasible_lambda = None
    feasible = False
    # loop over files in the space4ai-r folder
    space4air_path = os.path.join(application_dir, "space4ai-r")
    for filename in os.listdir(space4air_path):
        if filename.startswith("Output_max_Lambda_"):
            # open file and read the corresponding solution
            filepath = os.path.join(space4air_path, filename)
            with open(filepath, "r") as file:
                solution = json.load(file)
                # compare solutions
                existing_components = solution["components"]
                new_components = current_solution["components"]
                if coincide(existing_components, new_components):
                    highest_feasible_lambda = solution["Lambda"]
                    feasible = solution["feasible"]
                    break
    return highest_feasible_lambda, feasible


def get_max_number(system: dict, layer: str, resource: str) -> int:
    """
    Return the maximum number of available instances for a given resource
    """
    for resource_type in ["EdgeResources", "CloudResources"]:
        if resource_type in system:
            if layer in system[resource_type]:
                if resource in system[resource_type][layer]:
                    return system[resource_type][layer][resource]["number"]
    return -1


def increase_resources(application_dir: str, current_solution: dict) -> dict:
    """
    Update the given solution by setting the number of resource instances to 
    the maximum available number
    """
    improved_solution = current_solution
    # load the system description (or generate it if not available)
    system_file = os.path.join(application_dir, "space4ai-r/SystemFile.json")
    if not os.path.exists(system_file):
        parser = space4ai_parser.ParserYamlToJson(
            application_dir=application_dir, who="s4ai-r"
        )
        system_file = parser.make_system_file()
    with open(system_file, "r") as file:
        system = json.load(file)
    # loop over all components, deployments and partitions
    for c, component_data in current_solution["components"].items():
        for s, deployment_data in component_data.items():
            for h, partition_data in deployment_data.items():
                # get the selected computational layer (unique!)
                l = list(partition_data.keys())[0]
                # get the selected resource (unique!)
                r = list(partition_data[l].keys())[0]
                # get the maximum available number and update the solution
                max_n = get_max_number(system, l, r)
                if max_n > 0:   # -1 is returned for FaaS resources
                    improved_solution["components"][c][s][h][l][r][
                        "number"
                    ] = max_n
    return improved_solution


def check_feasibility(
      optimizer_config: dict, 
      optimizer_config_file: str
    ) -> bool:
    """
    Check the feasibility of a given production deployment under the given 
    workload value
    """
    # define space4ai-r optimizer executable
    optimizer = "s4ai-r-optimizer/BUILD/apps/check_feasibility_exe"
    # run
    command = f"{optimizer} {optimizer_config_file}"
    out = os.system(command)
    # if the run is successful, read the output
    if out == 0:
        solution_file = optimizer_config["BinarySearchOutputs"][0]
        with open(solution_file, "r") as file:
            feasible = json.load(file)["feasible"]
            # if the solution is feasible, copy it as new input
            if feasible:
                new_input_file = optimizer_config["DTSolutions"][0]
                shutil.copyfile(solution_file, new_input_file)
            return feasible
    return False


def binary_search(
      min_lambda: float, 
      max_lambda: float, 
      epsilon: float,
      optimizer_config: dict,
      optimizer_config_file: str,
      logger: space4ai_logger.Logger
    ) -> float:
    """
    Returns the maximum admissible workload for the given production 
    deployment, determined through a binary search
    """
    lowest_unfeasible_lambda = max_lambda
    highest_feasible_lambda = min_lambda
    current_lambda = min_lambda
    found_any_feasible = False
    tol = epsilon + 1   # let's be sure we enter the loop
    while tol > epsilon:
        # update the optimizer configuration file
        optimizer_config["Lambda"] = current_lambda
        with open(optimizer_config_file, "w") as ostream:
            json.dump(optimizer_config, ostream, indent=2)
        # check feasibility with the current workload
        logger.log(80*"-")
        logger.log(f"Check feasibility with workload {current_lambda}")
        currently_feasible = check_feasibility(
          optimizer_config=optimizer_config,
          optimizer_config_file=optimizer_config_file,
        )
        # compute next workload to check
        if currently_feasible:
            highest_feasible_lambda = current_lambda
            current_lambda = (lowest_unfeasible_lambda + current_lambda) / 2
            found_any_feasible = True
            logger.log(f"Feasible! --> next workload {current_lambda}")
        else:
            lowest_unfeasible_lambda = current_lambda
            current_lambda = (highest_feasible_lambda + current_lambda) / 2
            logger.log(f"Unfeasible --> next workload {current_lambda}")
        # update tolerance
        tol = abs(lowest_unfeasible_lambda - highest_feasible_lambda)
    return highest_feasible_lambda, found_any_feasible


def convert_verbosity_level(verbosity_level: str) -> int:
    """
    Return the verbosity level as required by the SPACE4AI-R-Optimizer 
    depending on the input string
    """
    if verbosity_level == "INFO":
        return 2
    elif verbosity_level == "DEBUG":
        return 1
    elif verbosity_level == "TRACE":
        return 0
    else:
        return -1


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------
@app.route(home_path, methods=["GET"])
def home():
    msg = f"Available! Post to {path} or {json_path}"
    return jsonify(msg), SUCCESS


@app.route(path, methods=["POST"])
def maximum_workload():
    """
    Compute the maximum admissible workload for the given configuration
    """
    data = request.get_json()
    max_workload = None
    # check existence of mandatory fields:
    KEY_ERROR = 0
    if "application_dir" not in data.keys():
        KEY_ERROR = 10
    else:
        application_dir = os.path.join(MOUNT_POINT, data["application_dir"])
        production_deployment_file = os.path.join(
            application_dir,
            "aisprint/deployments/current_deployment",
            "production_deployment.yaml"
        )
        # check that the application directory is accessible
        if not os.path.exists(application_dir):
            KEY_ERROR = 20
        # check that the production deployment file is accessible
        if not os.path.exists(production_deployment_file):
            KEY_ERROR = 30
        else:
            # initialize logger
            logger = space4ai_logger.Logger(name="SPACE4AI-R-MaxWorkloadApi")
            # get binary search parameters
            min_lambda = data.get("lowerBoundLambda")
            max_lambda = data.get("upperBoundLambda")
            epsilon = data.get("epsilon")
            if min_lambda is None or max_lambda is None or epsilon is None:
                lbl, ubl, eps = get_binary_search_parameters(application_dir)
                min_lambda = lbl if min_lambda is None else min_lambda
                max_lambda = ubl if max_lambda is None else max_lambda
                epsilon = eps if epsilon is None else epsilon
            logger.log(f"Binary search between {min_lambda} and {max_lambda}")
            # read production deployment
            production_deployment = {}
            with open(production_deployment_file, "r") as file:
                 production_deployment = yaml.full_load(file)["System"]
            # convert production deployment to the required format
            current_solution = change_format(
              application_dir=application_dir,
              production_deployment=production_deployment
            )
            logger.log(
                "Loaded current production deployment {}".format(
                    production_deployment_file
                )
            )
            # look for maximum workload in the already existing deployment(s)
            max_workload, found_any_feasible = read_max_workload(
              application_dir=application_dir, 
              current_solution=current_solution
            )
            if max_workload is None:
                logger.log("No pre-existing values for the given solution")
                # select the highest number of resources in the current 
                # solution
                current_solution = increase_resources(
                    application_dir, 
                    current_solution
                )
                logger.log("Resources increased to the maximum value")
                # write current solution to a file
                solution_file = os.path.join(
                    application_dir,
                    f"space4ai-r/solution_for_binary_search.json"
                )
                with open(solution_file, "w") as file:
                    json.dump(current_solution, file, indent=2)
                logger.log(f"Solution written at {solution_file}")
                # get verbosity level
                verbosity_level = convert_verbosity_level(
                    data.get("verbosity_level", "INFO")
                )
                # initialize the optimizer configuration file
                config = {
                    "ConfigFiles": [
                        os.path.join(
                            application_dir, "space4ai-r/SystemFile.json"
                        )
                    ],
                    "DTSolutions": [solution_file],
                    "BinarySearchOutputs": [
                        os.path.join(
                            application_dir,
                            "space4ai-r/check_feasibility_output.json"
                        )
                    ],
                    "Lambda": None,
                    "Logger": {
                        "priority": verbosity_level,
                        "terminal_stream": True,
                        "file_stream": False,
                    }
                }
                config_file = os.path.join(
                    application_dir, 
                    "space4ai-r/BinarySearchConfig.json"
                )
                # start binary search
                logger.log("Start binary search")
                max_workload, found_any_feasible = binary_search(
                  min_lambda=min_lambda,
                  max_lambda=max_lambda,
                  epsilon=epsilon,
                  optimizer_config=config,
                  optimizer_config_file=config_file,
                  logger=logger
                )
                logger.log(
                    "Binary search terminates with {} value {}".format(
                        "FEASIBLE" if found_any_feasible else "UNFEASIBLE",
                        max_workload
                    )
                )
                if found_any_feasible:
                    # save the final solution
                    deployment_name = production_deployment["DeploymentName"]
                    max_load_solution_file = os.path.join(
                        application_dir,
                        f"space4ai-r/Output_max_Lambda_{deployment_name}.json"
                    )
                    shutil.copyfile(solution_file, max_load_solution_file)
                    logger.log(
                        f"Final solution written at {max_load_solution_file}"
                    )
            else:
                logger.log(f"Found already existing value: {max_workload}")
            # define output
            output = (
              {"max_workload": max_workload, "feasible": found_any_feasible}, 
              SUCCESS
            )
    # if any key error is defined, return error code
    if KEY_ERROR > 0:
        output = (
          error_msg[NOT_FOUND + KEY_ERROR], 
          NOT_FOUND + KEY_ERROR
        )
    return jsonify(output[0]), output[1]


@app.route(json_path, methods=["POST"])
def maximum_workload_json():
    """
    Compute the maximum admissible workload for the given configuration
    """
    data = request.get_json()
    max_workload = None
    # check existence of mandatory fields:
    KEY_ERROR = 0
    if "application_dir" not in data.keys():
        KEY_ERROR = 10
    else:
        if "lowerBoundLambda" not in data.keys():
            KEY_ERROR = 40
        else:
            min_lambda = data["lowerBoundLambda"]
            # define directories and files paths
            application_dir = data["application_dir"]
            input_dir = os.path.join(MOUNT_POINT, "input", application_dir)
            s4air_output_dir = os.path.join(
                MOUNT_POINT, "output", application_dir, "space4air"
            )
            output_dir = os.path.join(
                MOUNT_POINT, "output", application_dir, "maxworkloadapi"
            )
            system_file = os.path.join(input_dir, "SystemFile.json")
            current_solution_file = os.path.join(
                s4air_output_dir, f"Lambda_{min_lambda}.json"
            )
            # check that the system file is accessible
            if not os.path.exists(system_file):
                KEY_ERROR = 50
            # check that the current solution file is accessible
            if not os.path.exists(current_solution_file):
                KEY_ERROR = 60
            else:
                # initialize logger
                logger = space4ai_logger.Logger()
                # get binary search parameters
                max_lambda = data.get("upperBoundLambda")
                epsilon = data.get("epsilon")
                if max_lambda is None or epsilon is None:
                    KEY_ERROR = 70
                else:
                    logger.log(
                        f"Binary search between {min_lambda} and {max_lambda}"
                    )
                    max_workload = min_lambda
                    found_any_feasible = False
                    # copy solution
                    solution_file = os.path.join(
                        output_dir, f"Lambda_{min_lambda}.json"
                    )
                    shutil.copyfile(
                        current_solution_file, solution_file
                    )
                    # get verbosity level
                    verbosity_level = convert_verbosity_level(
                        data.get("verbosity_level", "INFO")
                    )
                    # initialize the optimizer configuration file
                    config = {
                        "ConfigFiles": [system_file],
                        "DTSolutions": [solution_file],
                        "BinarySearchOutputs": [
                            os.path.join(
                                output_dir, 
                                "check_feasibility_output.json"
                            )
                        ],
                        "Lambda": None,
                        "Logger": {
                            "priority": verbosity_level,
                            "terminal_stream": True,
                            "file_stream": False,
                        }
                    }
                    config_file = os.path.join(
                        output_dir, 
                        "BinarySearchConfig.json"
                    )
                    # start binary search
                    logger.log("Start binary search")
                    max_workload, found_any_feasible = binary_search(
                        min_lambda=min_lambda,
                        max_lambda=max_lambda,
                        epsilon=epsilon,
                        optimizer_config=config,
                        optimizer_config_file=config_file,
                        logger=logger
                    )
                    logger.log(
                        "Binary search terminates with {} value {}".format(
                            "FEASIBLE" if found_any_feasible else "UNFEASIBLE",
                            max_workload
                        )
                    )
                    if found_any_feasible:
                        # save the final solution
                        max_load_solution_file = os.path.join(
                            output_dir,
                            f"Output_max_Lambda_{max_workload}.json"
                        )
                        shutil.move(
                            solution_file, max_load_solution_file
                        )
                        logger.log(
                            f"Final solution written at {max_load_solution_file}"
                        )
                    # define output
                    output = (
                        {
                            "max_workload": max_workload, 
                            "feasible": found_any_feasible
                        }, 
                        SUCCESS
                    )
    # if any key error is defined, return error code
    if KEY_ERROR > 0:
        output = (
            error_msg[NOT_FOUND + KEY_ERROR], 
            NOT_FOUND + KEY_ERROR
        )
    return jsonify(output[0]), output[1]


# ----------------------------------------------------------------------------
# start
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8008)
    # serve(app, host="0.0.0.0", port=8008)
