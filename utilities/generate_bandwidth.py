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
from typing import Tuple
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
      description="Bandwidth Trace Generation"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--min_bandwidth", 
      help="Minimum bandwidth value", 
      type=float
    )
    parser.add_argument(
      "--max_bandwidth", 
      help="Maximum bandwidth value", 
      type=float
    )
    parser.add_argument(
      "--max_steps", 
      help="Bandwidth trace length", 
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


class BandwidthGenerator:
  def __init__(
      self, 
      base_trace_file: str,
      num_shifts: int = 5,
      shift_range: Tuple[int,int] = (-1000, 1000),
      random_noise: float = 0.1,
      nRowsRead: int = None,
      logger: space4ai_logger.Logger = space4ai_logger.Logger(
        name = "BandwidthGenerator"
      )
    ) -> None:
    # load base trace
    df = pd.read_csv(base_trace_file, delimiter = ",", nrows = nRowsRead)
    self.base_trace = np.array(df["Throughput"].to_list())
    # initialize parameters
    self.num_shifts = num_shifts
    self.shift_range = shift_range
    self.random_noise = random_noise
    # initialize logger
    self.logger = logger
  
  def _random_shift_trace(
      self,
      throughput_values: np.array, 
      rng: np.random.Generator
    ) -> np.array:
    shift = rng.integers(self.shift_range[0], self.shift_range[1])
    shifted_trace = np.roll(throughput_values, shift)
    self.logger.log(f"  applied random shift of {shift} positions", v = 1)
    return shifted_trace

  def _multiple_random_shifts(
      self,
      throughput_values: np.array, 
      rng: np.random.Generator
    ) -> list:
    shifted_traces = []
    for _ in range(self.num_shifts):
      shifted = self._random_shift_trace(
        throughput_values, 
        rng
      )
      shifted_traces.append(shifted)
    return shifted_traces

  def _invert_trace_middle(self, throughput_values: np.array) -> np.array:
    n = len(throughput_values)
    middle_idx = n // 2
    # Create inverted trace by swapping halves
    # Second half (middle_idx to end) goes to beginning
    # First half (0 to middle_idx) goes to end
    inverted_trace = np.concatenate([
        throughput_values[middle_idx:],  # Second half to beginning
        throughput_values[:middle_idx]   # First half to end
    ])
    return inverted_trace

  def _extract_and_add_noise(
      self, 
      base_array: np.array, 
      size: int, 
      rng: np.random.Generator
    ) -> np.array:
    base_length = len(base_array)
    # Generate random integer indices
    indices = rng.integers(0, base_length, size = size)
    # Generate random percentage changes in a vectorized way
    percentage_changes = rng.uniform(
      -self.random_noise, self.random_noise, size = size
    )
    # Apply the random percentage changes to the selected elements
    random_values = base_array[indices] * (1 + percentage_changes)
    return np.round(random_values, 2)  # Round and return the new values
  
  def generate_trace(
      self, 
      max_len: int, 
      min_throughput: float, 
      max_throughput: float, 
      rng: np.random.Generator
    ) -> np.array:
    # generate additional traces by random shifts and inversion
    self.logger.log(f"Apply random shifts")
    additional_traces = self._multiple_random_shifts(self.base_trace, rng)
    self.logger.log(f"Invert trace")
    additional_traces.append(self._invert_trace_middle(self.base_trace))
    self.logger.log(f"Concatenate")
    full_trace = np.concatenate(
      (self.base_trace, np.concatenate(additional_traces))
    )
    # extract
    self.logger.log(f"Extract and add noise")
    bw = self._extract_and_add_noise(full_trace, max_len, rng)
    # rescale between min and max
    self.logger.log(f"Rescale")
    bw_min = bw.min()
    bw_max = bw.max()
    bw_scaled = np.array([
      rescale(b, bw_min, bw_max, min_throughput, max_throughput) for b in bw
    ])
    return bw_scaled

  def plot_traces_intervals(
      self,
      bw: np.array, 
      start: int = 20000, 
      steps: int = 2000, 
      current_step: int = 20000, 
      name: str = "5G_trace", 
      dpi: int = 100, 
      plot_folder: str = None
    ):
    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(figsize = (12,4))
    ax.plot(bw[start:start+steps], ".-")
    # highlight current time
    ax.axvline(
      x = current_step,
      color = "k",
      linestyle = "dashed"
    )
    # naming the x axis
    ax.set_xlabel("Control time period $t$")
    # naming the y axis
    ax.set_ylabel('Throughput(Mbps)')
    plt.tight_layout()
    plt.grid(True)
    if plot_folder is not None:
      plt.savefig(
        os.path.join(plot_folder, f"{name}_{start}_{start+steps}.png"),
        dpi = dpi,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()
    else:
      plt.title(f"{name}_{start}_{start+steps}")
      plt.show()


def main(
    application_dir: str, 
    min_bandwidth: float, 
    max_bandwidth: float, 
    max_steps: int, 
    seed: int, 
    logger: space4ai_logger.Logger
  ):
  # initialize generator
  BWG = BandwidthGenerator(
    os.path.join(application_dir, "5G_trace.csv"),
    num_shifts = 5,
    shift_range = (-1000, 1000), 
    random_noise = 0.1, 
    logger = logger
  )
  rng = np.random.default_rng(seed = seed)
  # generate trace
  bw_trace = BWG.generate_trace(max_steps, min_bandwidth, max_bandwidth, rng)
  # save
  os.makedirs(application_dir, exist_ok = True)
  with open(
    os.path.join(application_dir, "BandwidthValues.json"), "w"
  ) as istream:
    lv = {"BandwidthVec": [min_bandwidth] + bw_trace.tolist()}
    istream.write(json.dumps(lv, indent = 2))
  # plot trace
  BWG.plot_traces_intervals(
    bw_trace,
    start = 0,
    steps = max_steps,
    current_step = 0,
    name = "BandwidthValues",
    dpi = 300,
    plot_folder = application_dir
  )
   


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  application_dir = args.application_dir
  min_bandwidth = args.min_bandwidth
  max_bandwidth = args.max_bandwidth
  max_steps = args.max_steps
  seed = args.seed
  verbose = args.verbose
  # set seed for random number generation
  np.random.seed(seed)
  # run
  logger = space4ai_logger.Logger(name="GenerateBandwidth", verbose=verbose)
  main(application_dir, min_bandwidth, max_bandwidth, max_steps, seed, logger)
