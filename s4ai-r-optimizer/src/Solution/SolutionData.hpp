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
* \file SolutionData.hpp
*
* \brief Defines the class to store the main data of the Solution.
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#ifndef SOLUTION_DATA_HPP_
#define SOLUTION_DATA_HPP_

#include "src/TypeTraits.hpp"

namespace Space4AI
{
class Solution; // forward declaration

/** Class to store the main data of the Solution. */
class SolutionData
{
    friend class Solution;
    friend class LocalSearch;

  private:

    /** default constructor */
    SolutionData() = default;

    /** default copy constructor */
    SolutionData(const SolutionData&) = default;

    /** default move constructor */
    SolutionData(SolutionData&&) = default;

    /** default copy operator */
    SolutionData& operator=(const SolutionData&) = default;

    /** default move copy operator */
    SolutionData& operator=(SolutionData&&) = default;

  public:

    /** y_hat getter */
    const YHatType&
    get_y_hat() const { return y_hat; }

    /** used_resources getter */
    const UsedResourcesOrderedType&
    get_used_resources() const {return used_resources;}

    const std::pair<size_t, size_t>&
    get_first_cloud() const {return first_cloud; }

    /** n_used_resources getter */
    const UsedResourcesNumberType&
    get_n_used_resources() const {return n_used_resources;}

    /** default destructor */
    ~SolutionData() = default;

  private:

    /** Hyper_Matrix of dimension 4, storing the number of Resource of a specific type
    *   used to run a specific Partition of a specific Component.
    *
    *   See TypeTraits.hpp to understand the type structure
    */
    YHatType y_hat;

    /** Structure to store only the deployed resources.
    *
    *   For each Component (indexing the first vector) we save a vector of tuples,
    *   [Partition index, ResourceType index, Resource index] used on that Component.
    *   Actually it could be sufficient to use just y_hat, but since the number of
    *   chosen resources can be much smaller than available resources, it is better
    *   to have a data structure that keeps track only of the used resources.
    *
    *   The structure is ordered, in fact the inner vector must be ordered by
    * 	part_idx at some point.
    *   Using a std::set would not be efficient since order is not necessary to
    *   keep the order for each insert (See report for details)
    */
    UsedResourcesOrderedType used_resources;

    std::pair<size_t, size_t> first_cloud;

    /** For each [Resource Type, Resource idx] save the number of instances of
    *   such Resource deployed in the System.
    */
    UsedResourcesNumberType n_used_resources;

};

} //namespace Space4AI

#endif /* SOLUTION_DATA_HPP_ */
