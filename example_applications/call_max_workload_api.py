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
    parser = argparse.ArgumentParser(description="SPACE4AI-R max-load api")
    parser.add_argument(
      "--check_home",
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
      "--epsilon", 
      help="Binary search tolerance", 
      type=float,
      default=None
    )
    args, _ = parser.parse_known_args()
    return args


def main(args: argparse.Namespace):
    # get environment variables with url and port
    API_URL = os.getenv("S4AIR_MAXLOADAPI_URL", "0.0.0.0")
    API_PORT = os.getenv("S4AIR_MAXLOADAPI_PORT", "8008")
    # check home, if required
    if args.check_home:
        url = f"http://{API_URL}:{API_PORT}/"
        sample_result = requests.get(url = url)
        print(sample_result)
        print(sample_result.json())
    else:
        # define data
        sample_data = {
          "application_dir": args.application_dir
        }
        if args.min_load is not None:
            sample_data["lowerBoundLambda"] = args.min_load
        if args.max_load is not None:
            sample_data["upperBoundLambda"] = args.max_load
        if args.epsilon is not None:
            sample_data["epsilon"] = args.epsilon
        # send request
        url = f"http://{API_URL}:{API_PORT}/space4air/workload/json"
        sample_result = requests.post(url = url, json = sample_data)
        print(sample_result)
        print(sample_result.json())


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
