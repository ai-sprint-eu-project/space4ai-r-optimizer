# SPACE4AI-R Optimizer Utilities

This folder includes utilities for the `SPACE4AI-R-Optimizer` tool. It can 
be built as a separate container, since the utilities do not require 
running the `SPACE4AI-R-Optimizer`, but rather concern data generation and 
results post-processing.

## Deployment instructions

### Docker image generation

From the **base directory** `space4ai-r-optimizer`, run:

```
IMG_NAME=aisprint/space4ai-r
IMG_TAG=utilities
docker build -t ${IMG_NAME}:${IMG_TAG} -f utilities/Dockerfile utilities
```

### Start the container

To start the docker container in interactive mode, run:

```
CONT_NAME=s4airutilities
PATH_TO_VOLUME=${PWD}/example_applications
MOUNT_POINT=/mnt
docker run  -it \
            --name ${CONT_NAME} \
            -e MOUNT_POINT=${MOUNT_POINT} \
            -v ${PATH_TO_VOLUME}:${MOUNT_POINT} \
            ${IMG_NAME}:${IMG_TAG}
```

where `PATH_TO_VOLUME` is the (global) path to the volume where the 
application directory is stored.

## Execution instructions

### Generate a unique testing scenario

The [`generate_scenario`](generate_scenario.py) script can be used to 
automatically generate the system description file for a single testing 
scenario according to the parameters specified in a suitable configuration 
file. It can be called from the command-line with the following prototype:

```
usage: generate_scenario.py [-h] [--application_dir APPLICATION_DIR] 
                            [--seed SEED] [-v VERBOSE]

Testing Scenario Generation

options:
  -h, --help            show this help message and exit
  --application_dir APPLICATION_DIR
                        Path to the application directory
  --seed SEED           Seed for random number generation
  -v VERBOSE, --verbose VERBOSE
                        Verbosity level
```

The only mandatory parameter is `application_dir`, which denotes the path to 
the application directory. This must include a configuration file, provided 
in JSON format and named `config.json`, reporting the values of the parameters 
considered to generate the testing scenario. The structure of this 
configuration file is detailed in [the following](#single-testing-scenario).

At the end of the execution, the `application_dir` has the following structure:

```
.
├── Instance0
│   └── SystemFile.json
├── Instance1
│   └── SystemFile.json
├── [...]
├── [...]
├── Instance<I>
│   └── SystemFile.json
└── config.json
```

> [!NOTE]
> To ensure compatibility with the provided plotting script, it is recommended 
> to follow the directory structure reported 
> [below](#structure-of-the-results-folder). To achieve this goal, consider 
> the `Scenario<S>` as your `application_dir` here.

### Generate multiple testing scenarios

The [`generate_scenarios`](generate_scenarios.py) script can be used to 
automatically generate the system description file for a set of testing 
scenarios according to the parameters specified in a suitable configuration 
file. It can be called from the command-line with the following prototype:

```
usage: generate_scenarios.py  [-h] 
                            [--application_dir APPLICATION_DIR] 
                            [--lambda_max LAMBDA_MAX [LAMBDA_MAX ...]] 
                            [--bandwidth_min BANDWIDTH_MIN [BANDWIDTH_MIN ...]] 
                            [--seed SEED] 
                            [-v VERBOSE]

Testing Scenarios Generation

options:
  -h, --help            show this help message and exit
  --application_dir APPLICATION_DIR
                        Path to the application directory
  --lambda_max LAMBDA_MAX [LAMBDA_MAX ...]
                        Maximum workload values
  --bandwidth_min BANDWIDTH_MIN [BANDWIDTH_MIN ...]
                        Minimum bandwidth values
  --seed SEED           Seed for random number generation
  -v VERBOSE, --verbose VERBOSE
                        Verbosity level
```

The mandatory parameters are:
* `application_dir`, which denotes the path to the application directory. 
This must include a configuration file, provided in JSON format and named 
`base_config.json`, reporting the values of the parameters considered to 
generate the testing scenarios. The structure of this configuration file 
is detailed in [the following](#multiple-testing-scenarios).
* `lambda_max`, which corresponds to the (list of) maximum workload value(s) 
to be considered in the tests.
* `bandwidth_min`, which corresponds to the (list of) minimum bandwidth 
value(s) to be considered in the tests.

> [!WARNING]
> The `lambda_max` and `bandwidth_min` parameters are compatible if: (i) the 
> two lists have the same size, OR (ii) one of the two lists include a single 
> element (which is repeated in all experiments).

At the end of the execution, the `application_dir` includes several 
subdirectories, one for each pair of `LAMBDA_MAX` and `BANDWIDTH_MIN` passed 
as parameters, and (assuming a single pair of `LAMBDA_MAX` and 
`BANDWIDTH_MIN`) it has the following structure:

```
.
├── Lambda_<LAMBDA_MAX>-Bandwidth_<BANDWIDTH_MIN>
│   ├── Scenario0
│   │   ├── Instance0
│   │   │   └── SystemFile.json
│   │   ├── Instance1
│   │   │   └── SystemFile.json
│   │   ├── [...]
│   │   ├── [...]
│   │   ├── Instance<I>
│   │   │   └── SystemFile.json
│   │   └── config.json
│   ├── Scenario1
│   │   ├── Instance0
│   │   │   └── SystemFile.json
│   │   ├── Instance1
│   │   │   └── SystemFile.json
│   │   ├── [...]
│   │   ├── [...]
│   │   ├── Instance<I>
│   │   │   └── SystemFile.json
│   │   └── config.json
│   ├── [...]
│   ├── [...]
│   ├── Scenario<S>
│   │   ├── Instance0
│   │   │   └── SystemFile.json
│   │   ├── Instance1
│   │   │   └── SystemFile.json
│   │   ├── [...]
│   │   ├── [...]
│   │   ├── Instance<I>
│   │   │   └── SystemFile.json
│   │   └── config.json
└── base_config.json
```

### Generate workload and bandwidth traces

To generate an input workload trace with the required minimum and maximum 
values, run the [`generate_workload.py`](generate_workload.py) script as 
follows:

```
usage: generate_workload.py [-h] 
                            [--application_dir APPLICATION_DIR] 
                            [--min_load MIN_LOAD] 
                            [--max_load MAX_LOAD] 
                            [--max_steps MAX_STEPS] 
                            [--seed SEED] 
                            [-v VERBOSE]

Workload Trace Generation

options:
  -h, --help            show this help message and exit
  --application_dir APPLICATION_DIR
                        Path to the application directory
  --min_load MIN_LOAD   Minimum workload value
  --max_load MAX_LOAD   Maximum workload value
  --max_steps MAX_STEPS
                        Workload trace length
  --seed SEED           Seed for random number generation
  -v VERBOSE, --verbose VERBOSE
                        Verbosity level
```

In particular, consider as `application_dir` the base application directory 
containing all the `Scenario<S>` subfolders, i.e., the base directory with 
name `Lambda_<LAMBDA_MAX>-Bandwidth_<BANDWIDTH_MIN>`. Moreover, when 
providing the `min_load` and `max_load` input parameters, ensure that the 
latter matches the `LAMBDA_MAX` value in the directory name. According to 
the provided input, a base workload trace is generated with `max_steps` values 
in `[min_load,max_load]`.

Similarly, to generate a bandwidth trace, run the 
[generate_bandwidth.py](generate_bandwidth.py) script as:

```
usage: generate_bandwidth.py  [-h] 
                              [--application_dir APPLICATION_DIR] 
                              [--min_bandwidth MIN_BANDWIDTH] 
                              [--max_bandwidth MAX_BANDWIDTH] 
                              [--max_steps MAX_STEPS] 
                              [--aggregate_every AGGREGATE_EVERY] 
                              [--seed SEED] 
                              [-v VERBOSE]

Bandwidth Trace Generation

options:
  -h, --help            show this help message and exit
  --application_dir APPLICATION_DIR
                        Path to the application directory
  --min_bandwidth MIN_BANDWIDTH
                        Minimum bandwidth value
  --max_bandwidth MAX_BANDWIDTH
                        Maximum bandwidth value
  --max_steps MAX_STEPS
                        Bandwidth trace length
  --aggregate_every AGGREGATE_EVERY
                        Interval length for average
  --seed SEED           Seed for random number generation
  -v VERBOSE, --verbose VERBOSE
                        Verbosity level
```

The parameters follow the same logic as above.

> [!NOTE]
> The bandwidth trace is generated based on real 5G values stored in the 
> [`5G_trace.csv`](base_data/5G_trace.csv) file, which is available to 
> the [`s4airutilities` container](#start-the-container) as 
> `/home/SPACE4AI-R-utilities/base_data/5G_trace.csv`.

> [!CAUTION]
> To match a use-case where SPACE4AI-D runs first considering the worst-case 
> scenario (i.e., the maximum expected workload and minimum expected 
> bandwidth), the first value in the generated traces corresponds to this 
> worst-case value. Ignore it if you want to consider only the dynamic trace 
> for SPACE4AI-R.

### Plot comparative results

The [`plot_results`](plot_results.py) script can be used to plot and compare 
the results achieved with the SPACE4AI-R Optimizer, exploiting the available 
heuristics. It can be called from the command-line with the following 
prototype:

```
usage: plot_results.py  [-h] [--results_dir RESULTS_DIR] 
                        [--heuristics HEURISTICS [HEURISTICS ...]] 
                        [--target TARGET]
                        [--skip_instance] 
                        [--skip_scenario] 
                        [--skip_workload]

Results Postprocessing

options:
  -h, --help            show this help message and exit
  --results_dir RESULTS_DIR
                        Path to the results directory
  --heuristics HEURISTICS [HEURISTICS ...]
                        List of heuristics to consider (available: s4air, uheur)
  --target TARGET       Target method
  --skip_instance       True if instance-specific plots should not be generated
  --skip_scenario       True if scenario-specific plots should not be generated
  --skip_workload       True if workload-specific plots should not be generated
```

The mandatory parameters are:
* `results_dir`, which denotes the path to the results directory. 
This should have the structure described in 
[the following](#structure-of-the-results-folder).
* `heuristics`, which corresponds to the (list of) heuristic method(s) 
to be validated and compared.

In particular, the comparison is performed, by default, with respect to 
the base implementation of the SPACE4AI-R Optimizer, denoted by `s4air`. 
The `target` method can be modified by providing a new value to the 
corresponding parameter.

## Structure of configuration files

This section defines the fields to be included in the configuration files 
used to generate the problem instances to be considered in one or multiple 
testing scenarios. These will be processed by the corresponding Python scripts 
defined in the previous sections.

### Single testing scenario

The `generate_scenario` script described 
[above](#generate-a-unique-testing-scenario) will produce `n_instances` random 
problem instances with the following characteristics:
* `n_components` application components, with:
  * between 1 and `max_n_deployments` candidate deployments for each component
  * between 1 and `max_n_partitions` partitions for each candidate deployment
  * if `linear_pipeline` is `true`, the components DAG is linear (no branches)
* if the `edge_resources` dictionary is defined,
  * `n_layers` computational layers including Edge devices
  * between 1 and `max_n_resources` Edge devices in each layer, with:
    * a number of resource instances between `min_n_instances` and 
    `max_n_instances`
    * a memory capacity between `memory` and `memory` multiplied by the total 
    number of application components
    * a per-second cost between `min_cost` and `max_cost`
    * a number of cores between 1 and `max_n_cores`
* if the `cloud_resources` dictionary is defined,
  * `n_layers` computational layers including Cloud VMs, with the same set of 
  characteristics detailed in the Edge case
* if the `faas_resources` dictionary is defined,
  * one computational layer including `n_faas_per_component` FaaS instances 
  for each application component, characterized by:
    * a memory capacity between `memory` and `memory` multiplied by the total 
    number of application components
    * a per-second cost between `min_cost` and `max_cost`
    * an `idle_time_before_kill` equal to 600s
  * a `transition_cost` equal to 0.0

Furthermore, the configuration file allows to specify:
* `max_n_candidates`: the maximum number of candidate resources for each 
component partition. **Note:** exactly `max_n_candidates` candidate resources 
are assigned to each component partition, unless the total number of 
generated resources does not allow to do so
* `allow_colocation`: whether multiple partitions can share the same candidate 
resource (namely, component co-location is allowed)
* `n_only_edge_components`: the number of components that can be executed 
only on Edge candidate resources
* `n_only_cloud_components`: the number of components that can be executed 
only on Cloud candidate resources
* `edge_demand_range`: the minimum and maximum demand of component partitions 
executed on Edge resources
* `cloud_demand_range`: the minimum and maximum demand of component partitions 
executed on Cloud resources
* `faas_warm_demand_range`: the minimum and maximum demand of warm requests 
on FaaS resources
* `faas_cold_demand_increase`: the percentage to increase the demand in the 
case of cold requests 
* `edge_memory_values`: TO BE FIXED
* `cloud_memory_values`: TO BE FIXED
* `faas_memory_values`: TO BE FIXED
* `data_size_range`: the minimum and maximum value of transferred data
* `network_technology`: a list of network domains and corresponding 
characteristics; each domain is represented by:
  * the list of computational layers included in the domain, or `"all"` if 
  all layers should be considered
  * the access delay
  * the bandwidth
* `max_n_local_constraints`: the maximum number of local constraints
* `local_threshold_range`: a range used to generate the local constraints 
thresholds (defined as the average demand of the considered component, 
multiplied by a value extracted randomly from the given range)
* `max_n_global_constraints`: the maximum number of global constraints
* `constrain_whole_application`: whether a global constraint on the entire 
application should be considered
* `global_threshold_range`: a range used to generate the global constraints 
thresholds (defined as the average demand of the considered path, 
multiplied by a value extracted randomly from the given range)
* `time`: time horizon
* `lambda`: input workload value

The types of all inputs are listed in the following:

```
{
  "n_instances": int,
  "n_components": int,
  "linear_pipeline": bool,
  "max_n_deployments": int,
  "max_n_partitions": int,
  "edge_resources": {
    "n_layers": int,
    "max_n_resources": int,
    "min_n_instances": int,
    "max_n_instances": int,
    "memory": int,
    "min_cost": float,
    "max_cost": float,
    "max_n_cores": int
  },
  "cloud_resources": {
    "n_layers": int,
    "max_n_resources": int,
    "min_n_instances": int,
    "max_n_instances": int,
    "memory": int,
    "min_cost": float,
    "max_cost": float,
    "max_n_cores": int
  },
  "faas_resources": {
    "n_faas_per_component": int,
    "memory": int,
    "min_cost": float,
    "max_cost": float
  },
  "max_n_candidates": int,
  "allow_colocation": bool,
  "n_only_edge_components": int,
  "n_only_cloud_components": int,
  "n_only_faas_components": int,
  "edge_demand_range": [float, float],
  "cloud_demand_range": [float, float],
  "faas_warm_demand_range": [float, float],
  "faas_cold_demand_increase": float,
  "edge_memory_values": [int, int],
  "cloud_memory_values": [int, int],
  "faas_memory_values": [int, int],
  "data_size_range": [int, int],
  "network_technology": [[[int, ...], float, int]] or [["all", float, int]],
  "max_n_local_constraints": int,
  "local_threshold_range": [float, float],
  "max_n_global_constraints": int,
  "global_threshold_range": [float, float],
  "constrain_whole_application": bool,
  "time": int,
  "lambda": float
}
```

### Multiple testing scenarios

The configuration file to generate multiple testing scenarios allows to 
specify a grid of parameter values to be considered. It is defined in JSON 
format and includes two dictionaries:
* `base_config`: a sub-dictionary with the same structure specified in the 
previous section, including the subset of parameters that should have a 
fixed value in all the scenarios
* `parameters_grid`: a sub-dictionary including the parameters whose values 
should vary in the considered scenarios. The keys should be the same as 
defined in the previous section, and the values to explore should be provided 
as lists. All the lists should have either the same length or length equal to 
1 (in which case, the same value will be considered in all scenarios): the 
n-th scenario is generated extracting from all lists the parameter value in 
position n, no combinatorial exploration is considered.

## Structure of the results folder

```
.
├── <base application directory>
│   ├── Lambda_<LAMBDA_MAX>-Bandwidth_<BANDWIDTH_MIN>
│   │   ├── LambdaValues.json
│   │   ├── BandwidthValues.json
│   │   ├── Scenario0
│   │   │   ├── Instance0
│   │   │   │   ├── LambdaValues.json
│   │   │   │   ├── BandwidthValues.json
│   │   │   │   ├── Solution-lambda_<LAMBDA_MAX>-bandwidth_<BANDWIDTH_MIN>.json
│   │   │   │   ├── SystemFile.json
│   │   │   │   ├── s4air
│   │   │   │   ├── [...]
│   │   │   │   └── <other heuristics>
│   │   │   │       └── <heuristic comparison identifier>
│   │   │   ├── [...]
│   │   │   ├── Instance<I>
│   │   │   │   ├── LambdaValues.json
│   │   │   │   ├── BandwidthValues.json
│   │   │   │   ├── Solution-lambda_<LAMBDA_MAX>-bandwidth_<BANDWIDTH_MIN>.json
│   │   │   │   ├── SystemFile.json
│   │   │   │   ├── s4air
│   │   │   │   ├── [...]
│   │   │   │   └── <other heuristics>
│   │   │   │       └── <heuristic comparison identifier>
│   │   │   └── logs
│   │   │   │   ├── compare_heuristics_0.log
│   │   │   │   ├── compare_heuristics_1.log
│   │   │   │   ├── [...]
│   │   │   │   └── compare_heuristics_<I>.log
│   │   │   └── config.json
│   │   ├── [...]
│   │   ├── [...]
│   │   ├── Scenario<S>
│   │   │   ├── Instance0
│   │   │   │   ├── LambdaValues.json
│   │   │   │   ├── BandwidthValues.json
│   │   │   │   ├── Solution-lambda_<LAMBDA_MAX>-bandwidth_<BANDWIDTH_MIN>.json
│   │   │   │   ├── SystemFile.json
│   │   │   │   ├── s4air
│   │   │   │   ├── [...]
│   │   │   │   └── <other heuristics>
│   │   │   │       └── <heuristic comparison identifier>
│   │   │   ├── [...]
│   │   │   ├── Instance<I>
│   │   │   │   ├── LambdaValues.json
│   │   │   │   ├── BandwidthValues.json
│   │   │   │   ├── Solution-lambda_<LAMBDA_MAX>-bandwidth_<BANDWIDTH_MIN>.json
│   │   │   │   ├── SystemFile.json
│   │   │   │   ├── s4air
│   │   │   │   ├── [...]
│   │   │   │   └── <other heuristics>
│   │   │   │       └── <heuristic comparison identifier>
│   │   │   └── logs
│   │   │   │   ├── compare_heuristics_0.log
│   │   │   │   ├── compare_heuristics_1.log
│   │   │   │   ├── [...]
│   │   │   │   └── compare_heuristics_<I>.log
│   │   │   └── config.json
│   └── base_config.json
```
