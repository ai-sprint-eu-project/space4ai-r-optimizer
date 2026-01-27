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

import requests
import argparse
import os

def parse_arguments() -> argparse.Namespace:
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
      description="SPACE4AI-R check feasibility boundaries api"
    )
    parser.add_argument(
      "--check_home",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--aisprint",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--application_dir", 
      help="Path to the application directory", 
      type=str
    )
    parser.add_argument(
      "--min_load", 
      help="Lower bound of the binary search", 
      type=float,
      default=None
    )
    parser.add_argument(
      "--max_load", 
      help="Upper bound of the binary search", 
      type=float,
      default=None
    )
    parser.add_argument(
      "--min_bandwidth", 
      help="Lower bound of the binary search", 
      type=float,
      default=None
    )
    parser.add_argument(
      "--max_bandwidth", 
      help="Upper bound of the binary search", 
      type=float,
      default=None
    )
    parser.add_argument(
      "--epsilon", 
      help="Binary search tolerance", 
      type=float,
      default=None
    )
    parser.add_argument(
      "--method", 
      help="Method used to compute the solution", 
      type=str,
      default="space4air"
    )
    parser.add_argument(
      "--verbosity_level", 
      help="Verbosity level", 
      type=str,
      default="INFO"
    )
    args, _ = parser.parse_known_args()
    return args


def main(
        check_home: bool,
        application_dir: str,
        verbosity_level: str,
        min_load: float,
        max_load: float,
        min_bandwidth: float,
        max_bandwidth: float,
        epsilon: float,
        method: str,
        aisprint: bool
    ) -> dict:
    # get environment variables with url and port
    API_URL = os.getenv("S4AIR_MAXLOADAPI_URL", "0.0.0.0")
    API_PORT = os.getenv("S4AIR_MAXLOADAPI_PORT", "8008")
    # check home, if required
    sample_result = None
    if check_home:
        url = f"http://{API_URL}:{API_PORT}/"
        sample_result = requests.get(url = url)
        print(sample_result)
        print(sample_result.json())
    else:
        # define data
        sample_data = {
          "application_dir": application_dir,
          "verbosity_level": verbosity_level,
          "method": method
        }
        if min_load is not None:
            sample_data["lowerBoundLambda"] = min_load
        if max_load is not None:
            sample_data["upperBoundLambda"] = max_load
        if min_bandwidth is not None:
            sample_data["lowerBoundBandwidth"] = min_bandwidth
        if max_bandwidth is not None:
            sample_data["upperBoundBandwidth"] = max_bandwidth
        if epsilon is not None:
            sample_data["epsilon"] = epsilon
        # send request
        url = f"http://{API_URL}:{API_PORT}/space4air/checkfeasibility"
        if not aisprint:
            url += "/json"
        sample_result = requests.post(url = url, json = sample_data)
        print(sample_result)
        print(sample_result.json())
    return sample_result.json()


if __name__ == "__main__":
    args = parse_arguments()
    check_home = args.check_home
    application_dir = args.application_dir
    verbosity_level = args.verbosity_level
    min_load = args.min_load
    max_load = args.max_load
    min_bandwidth = args.min_bandwidth
    max_bandwidth = args.max_bandwidth
    epsilon = args.epsilon
    method = args.method
    aisprint = args.aisprint
    main(
        check_home,
        application_dir,
        verbosity_level,
        min_load,
        max_load,
        min_bandwidth,
        max_bandwidth,
        epsilon,
        method,
        aisprint
    )
