/*
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
*/

/**
* \file EliteResult.hpp
*
* \brief Defines the class to store to store a fixed-size vector of Solution objects, sorted by cost
*        (from least expensive to most expensive).
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#ifndef ELITE_RESULT_HPP_
#define ELITE_RESULT_HPP_

#include "src/Solution/Solution.hpp"
#include "src/System/System.hpp"

#include <algorithm>

namespace Space4AI
{
/** Class to store to store a fixed-size list of Solution objects, sorted by cost
*   (from least expensive to most expensive).
*/
class EliteResult
{
  public:

    /** EliteResult constructor
    *
    *   \param max_num_sols_ maximum length admissible for the vector
    */
    explicit EliteResult(size_t max_num_sols_): max_num_sols(max_num_sols_), num_threads(1)
    {
      solutions.reserve(max_num_sols_);
    }

    /** Print a solution to file.
    *
    *   \param system System to which the solutions refer
    *   \param path Path where to save the Solution .json configuration file
    *   \param rank Rank of the solution to be printed (0 the best, 1 the second best, ...)
    */
    void print_solution(const System& system, const std::string& path, size_t rank = 0) const;

    /** Method to add a Solution to the class EliteResult.
    *
    *   \param solution Solution object to be added
    */
    template<class T>
    void add(T&& solution);

    template<class T>
    void add_vec(std::vector<T>&& solutions_vec);

    /** solutions getter */
    const std::vector<Solution>&
    get_solutions() const { return solutions; }

    /** Method to get the current number of saved solutions */
    size_t
    get_size() const { return solutions.size(); }

    /** num_threads setter */
    void
    set_num_threads(size_t num_threads_) { num_threads = num_threads_; }

    /** num_threads getter */
    size_t
    get_num_threads() const { return num_threads; }

  private:

    /** Maximum number of solutions to save */
    const size_t max_num_sols;

    /** Vector to store the candidate solutions */
    std::vector<Solution> solutions;

    /** Number of threads used in the algorithm to find the solutions */
    size_t num_threads;

};

// TEMPLATE DEFINITIONS
template<class T>
void
EliteResult::add(T&& solution)
{
  solutions.push_back(std::forward<T>(solution));
  std::sort(solutions.begin(), solutions.end());
  solutions.resize(std::min(solutions.size(), max_num_sols), Solution(nullptr));
}

template<class T>
void
EliteResult::add_vec(std::vector<T>&& solutions_vec)
{
  solutions.insert(solutions.end(), solutions_vec.begin(), solutions_vec.end());
  std::sort(solutions.begin(), solutions.end());
  solutions.resize(std::min(solutions.size(), max_num_sols), Solution(nullptr));
}

} //namespace Space4AI

#endif /* ELITE_RESULT_HPP_ */
