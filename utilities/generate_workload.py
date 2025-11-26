"""
Copyright 2025 AI-SPRINT

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
from utilities_functions import rescale

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
      description="Workload Trace Generation"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--min_load", 
      help="Minimum workload value", 
      type=float
    )
    parser.add_argument(
      "--max_load", 
      help="Maximum workload value", 
      type=float
    )
    parser.add_argument(
      "--max_steps", 
      help="Workload trace length", 
      type=int,
      default=100
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


class LoadGenerator:
  def __init__(
      self, 
      # default values are calculated to match the average mean of the real
      # traces
      average_requests: int = 50, 
      amplitude_requests: int = 100,
      noise_ratio: float = 0.1,
      unique_periods: int = 3
    ) -> None:
    self.average_requests = average_requests
    self.amplitude_requests = amplitude_requests
    self.noise_ratio = noise_ratio
    self.unique_periods = unique_periods
  
  def _impose_system_workload(
      self, total_workload, input_requests: dict
    ) -> dict:
    # convert base input requests dict into a matrix
    base_signals = np.array([
      input_requests[agent] for agent in input_requests
    ])
    num_agents, num_timesteps = base_signals.shape
    # if a single value of total_workload is provided, it is the same at all
    # time steps
    if isinstance(total_workload, (int, float)):
      total_workload = np.full(num_timesteps, round(total_workload, 3))
    # normalize so that at each timestep, sum of all agents' 
    # workloads == total_workload[t]
    workloads = np.zeros_like(base_signals)
    for t in range(num_timesteps):
      total = np.sum(base_signals[:, t])
      if total == 0:
        workloads[:, t] = round(total_workload[t] / num_agents, 3)
      else:
        workloads[:, t] = [
          round(w, 3) for w in base_signals[:, t] / total * total_workload[t]
        ]
    # extract dictionary
    workloads_dict = {agent: workloads[agent] for agent in input_requests}
    return workloads_dict
  
  def _synthetic_sinusoidal_input_requests(
      self, 
      max_steps: int, 
      agents: list, 
      rng: np.random.Generator,
      only_integer_values: bool = False
    ) -> dict:
    """
    Generates the input requests for the given agents with the given length,
    clipping the values within the given bounds and using the given rng to
    generate the synthesized data.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    Returns a dictionary whose keys are the agent IDs and whose value is an
    np.ndarray containing the input requests for each step.
    """
    # generate
    input_requests = {}
    steps = np.arange(max_steps)
    for agent in agents:
      # Note: with default max_stes, the period changes every 96 steps
      # (max_steps = 288). We first generate the periods and expand the array
      # to match the max_steps. If max_steps is not a multiple of 96, some
      # elements must be appended at the end, hence the resize call.
      repeats = max_steps // self.unique_periods
      periods = rng.uniform(15, high = 100, size = self.unique_periods)
      periods = np.repeat(periods, repeats)  # Expand the single values.
      periods = np.resize(periods, periods.size + max_steps - periods.size)
      # base + noise
      base_input = self.average_requests + self.amplitude_requests * np.sin(
        2 * np.pi * steps / periods
      )
      noisy_input = base_input + self.noise_ratio * rng.normal(
        0, self.amplitude_requests, size=max_steps
      )
      requests = None
      if only_integer_values:
        requests = np.asarray(noisy_input, dtype = np.int32)
      else:
        requests = np.asarray(noisy_input)
      # save
      input_requests[agent] = requests
    return input_requests
  
  def generate_traces(
      self, 
      max_steps: int, 
      limits: dict, 
      rng: np.random.Generator, 
      trace_type: str = "clipped",
      only_integer_values: bool = False
    ):
    input_requests = self._synthetic_sinusoidal_input_requests(
      max_steps, limits.keys(), rng, only_integer_values
    )
    if trace_type in ["clipped", "sinusoidal"]:
      # Ensure the number of requests stays in [min, max]
      for agent, requests in input_requests.items():
        minr = limits[agent]["min"]
        maxr = limits[agent]["max"]
        if trace_type == "clipped":
          # Clip the excess values respecting the minimum and maximum values
          # for the input requests observation.
          np.clip(requests, minr, maxr, out = requests)
        elif trace_type == "sinusoidal":
          # Rescale
          in_min = requests.min()
          in_max = requests.max()
          if only_integer_values:
            requests = np.array([
              int(rescale(r, in_min, in_max, minr, maxr)) for r in requests
            ])
          else:
            requests = np.array([
              round(rescale(r, in_min, in_max, minr, maxr), 3) for r in requests
            ])
        input_requests[agent] = requests
    elif trace_type.startswith("fixed_sum"):
      total_workload = 0.0
      if trace_type == "fixed_sum":
        total_workload = sum(list(limits.values()))/len(list(limits.values()))
      else:
        total_workload = min(list(limits.values())) if trace_type.endswith(
          "min"
        ) else max(list(limits.values()))
      if isinstance(total_workload, dict):
        total_workload = total_workload["max"]
      input_requests = self._impose_system_workload(
        total_workload, input_requests
      )
    else:
      raise KeyError(f"Trace type `{trace_type}` is not supported")
    # ensure that everything, anyway, stays above zero
    for agent in input_requests:
      input_requests[agent] = np.array([
        max(0, r) for r in input_requests[agent]
      ])
    return input_requests

  @staticmethod
  def plot_input_load(
      input_requests: dict, current_step: int = 0, plot_filename: str = None
    ):
    _, ax = plt.subplots()
    for agent, incoming_load in input_requests.items():
      max_steps = len(incoming_load)
      ax.plot(
        range(max_steps),
        incoming_load,
        ".-",
        label = f"agent {agent}"
      )
    # highlight current time
    ax.axvline(
      x = current_step,
      color = "k",
      linestyle = "dashed"
    )
    # axis properties
    ax.set_xlabel("Control time period $t$", fontsize = 14)
    ax.set_ylabel("Load [req/s]", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.grid()
    if plot_filename is not None:
      plt.savefig(
        plot_filename, dpi = 300, format = "png", bbox_inches = "tight"
      )
      plt.close()
    else:
      plt.show()


def main(
    application_dir: str, 
    min_load: float, 
    max_load: float, 
    max_steps: int, 
    seed: int, 
    logger: space4ai_logger.Logger
  ):
  # initialize generator
  LG = LoadGenerator(average_requests = 100, amplitude_requests = 50)
  rng = np.random.default_rng(seed = seed)
  # generate trace
  limits = {0: {"min": min_load, "max": max_load}}
  load_trace = LG.generate_traces(
    max_steps = max_steps, 
    limits = limits, 
    rng = rng,
    trace_type = "sinusoidal",
    only_integer_values = False
  )
  # save trace
  os.makedirs(application_dir, exist_ok = True)
  with open(
    os.path.join(application_dir, "LambdaValues.json"), "w"
  ) as istream:
    lv = {"LambdaVec": [max_load] + load_trace[0].tolist()}
    istream.write(json.dumps(lv, indent = 2))
  # plot trace
  LG.plot_input_load(
    load_trace, 
    plot_filename = os.path.join(application_dir, "LambdaValues.png")
  )


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  application_dir = args.application_dir
  min_load = args.min_load
  max_load = args.max_load
  max_steps = args.max_steps
  seed = args.seed
  verbose = args.verbose
  # set seed for random number generation
  np.random.seed(seed)
  # run
  logger = space4ai_logger.Logger(name="GenerateWorkload", verbose=verbose)
  main(application_dir, min_load, max_load, max_steps, seed, logger)
