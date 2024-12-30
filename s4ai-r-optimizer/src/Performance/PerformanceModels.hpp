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
* \file PerformanceModels.hpp
*
* \brief Defines the classes for the performance evaluators.
*
* Polymorphism is used. Note that some Performance Models constructors
* could accept parameters that will not be used by the class. This trick is
* required to provide the same interface to the object factory builder
* (see PerformanceFactory.hpp). a-ML library predictors are used as pybind11 classes,
* implemented with the Meyer's singleton pattern in the file PerformancePredictors.hpp.
*
* \author Randeep Singh
* \author Giulia Mazzilli
*/

#ifndef PERFORMANCEMODELS_HPP_
#define PERFORMANCEMODELS_HPP_

#include "src/Performance/PerformancePredictors.hpp"
#include "src/Solution/SolutionData.hpp"
#include "src/System/SystemData.hpp"
#include "src/TypeTraits.hpp"

namespace Space4AI
{
/** Abstract Parent class of the performance models.
*
*   Abstract class used to represent a performance model for predicting the
*   response time of a Partition object deployed onto a given Resource.
*/
class BasePerformanceModel
{
  public:

    /** BasePerformanceModel constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    */
    BasePerformanceModel(
      const std::string& keyword_,
      bool allows_colocation_
    ): keyword(keyword_), allows_colocation(allows_colocation_)
    {}

    /** keyword getter */
    std::string
    get_keyword() const {return keyword;}

    /** allows_colocation getter */
    bool
    get_allows_colocation() const {return allows_colocation;}

    /** Method to evaluate the performance of a specific Partition object 
    *   executed onto a specific Resource.
    *
    *   \param comp_idx Component index
    *   \param part_idx Partition index in the component
    *   \param res_type Type of the Resource
    *   \param res_idx Resource index
    *   \param system_data Reference to all the SystemData read from the 
    *                      .json configuration file
    *   \param solution_data Reference to the SolutionData, namely 
    *                        SolutionData.y_hat and SolutionData.used_resources
    *   \return response time of the execution
    *
    *   The method will be overridden by the performance classes
    */
    virtual
    TimeType predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const = 0;

    /** virtual destructor */
    virtual ~BasePerformanceModel() = default;

    bool support_meanTime_usage() const {return meanTime_usage_supported;}

  protected:

    /** Name of the model */
    const std::string keyword;

    /** Boolean variable to determine whether colocation is allowed or not */
    const bool allows_colocation;

    /** Boolean variable to determine whether the average job execution time 
     * can be used to avoid the pipeline effect */
    const bool meanTime_usage_supported = false;
};

/** Class to define queue-servers performance models for Edge and VM. */
class QTPE: public BasePerformanceModel
{
  public:

    /** QTPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demand_ Demand time
    */
    QTPE(
      const std::string& keyword_,
      bool allows_colocation_
    ):
      BasePerformanceModel(keyword_, allows_colocation_)
    {}

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    TimeType predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const override;

    /** Method to compute the utilization of a specific Resource object.
    *
    *   \param res_type Type of the Resource
    *   \param res_idx Resource index
    *   \param system_data Reference to all the SystemData read from the .json 
    *                      configuration file
    *   \param solution_data Reference to the SolutionData, namely 
    *                        SolutionData.y_hat and SolutionData.used_resources
    *   \return utilization of the Resource
    */
    double compute_utilization(
      ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const;

    /** all_demands setter */
    template<class T>
    static
    void
    set_all_demands(T&& all_demands_)
    {
      QTPE::all_demands = std::forward<T>(all_demands_);
    }

    /** virtual destructor */
    virtual ~QTPE() = default;

  private:

    /** Demand times of all Component-Partition objects that could be run on
    *   the specific Resource of type Edge or Cloud VM.
    */
    inline
    static
    DemandEdgeVMType all_demands = {};

};

/** Abstract class inherited from BasePerformanceModel, to represent generic
*   FaaS performance models.
*/
class FaasPE: public BasePerformanceModel
{
  public:

    /** FaasPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demandWarm_ Demand time when there is an active server available 
    *                      on the platform
    *   \param demandCold_ Demand time when all the servers on the platform 
    *                      are down and a new one needs to be activated
    */
    FaasPE(
      const std::string& keyword_,
      bool allows_colocation_,
      TimeType demandWarm_,
      TimeType demandCold_
    ):
      BasePerformanceModel(keyword_, allows_colocation_),
      demandWarm(demandWarm_), demandCold(demandCold_)
    {}

    /** demandWarm getter */
    virtual
    TimeType
    get_demandWarm() const {return demandWarm; }

    /** demandCold getter */
    virtual
    TimeType
    get_demandCold() const {return demandCold; }

    /** virtual destructor */
    virtual ~FaasPE() = default;

  protected:

    /** Demand time when there is at least one active server available on 
     * the platform 
    */
    const TimeType demandWarm;

    /** Demand time when all the servers on the platform are down and a new 
     * one needs to be activated 
    */
    const TimeType demandCold;

};

/**
*   Class to define FaaS pacsltk performance model (dynamic version)
*
*   Class inherited from FaasPE. This class is very similar to 
*   FaasPacsltkStaticPE, with the crucial difference that this is used where 
*   there is the need to compute the demand time during run time, for instance 
*   if a workload variations has happened. In principle, it is not needed 
*   at Design Time.
*/
class FaasPacsltkPE: public FaasPE
{
  public:
    /** FaasPacsltkPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demandWarm_ Demand time when there is an active server 
    *                      available on the platform
    *   \param demandCold_ Demand time when all the servers on the platform 
    *                      are down and a new one needs to be activated
    */
    FaasPacsltkPE(
      const std::string& keyword_,
      bool allows_colocation_,
      TimeType demandWarm_,
      TimeType demandCold_
    ):
      FaasPE(keyword_, allows_colocation_, demandWarm_, demandCold_),
      predictor(Pacsltk::Instance())
    {}

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const override;

    virtual ~FaasPacsltkPE() = default;

  private:

    /** Instance of the singleton class Pacsltk used to compute the demand time
    *   through the use of the Python library "pacsltk"*/
    Pacsltk& predictor;

};
#pragma GCC diagnostic pop


/** Class to define FaaS pacsltk performance model (static version).
*
*   Class inherited from FaasPE. This class is very similar to FaasPacsltkPE, 
*   but it computes the demand time once and for all, at construction time. 
*   Thus, when the demand time is requested, the stored value is returned, 
*   without the need of recoumputing the same value. Obviously, this model 
*   can be used only under the assumption that the workload does not vary, 
*   which is satisfied for design time problems.
*/
class FaasPacsltkStaticPE: public FaasPE
{
  public:

    /** FaasPacsltkStaticPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demandWarm_ Demand time when there is an active server 
    *                      available on the platform
    *   \param demandCold_ Demand time when all the servers on the platform 
    *                      are down and a new one needs to be activated
    *   \param idle_time_before_kill How long does the platform keep the 
    *                                servers up after being idle
    *   \param part_lambda Load factor of the Partition
    */
    FaasPacsltkStaticPE(
      const std::string& keyword_,
      bool allows_colocation_,
      TimeType demandWarm_,
      TimeType demandCold_,
      TimeType idle_time_before_kill,
      LoadType part_lambda
    );

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data,
      const SolutionData& solution_data
    ) const override;

    /** virtual destructor */
    virtual ~FaasPacsltkStaticPE() = default;

  protected:

    /** Demand time */
    TimeType demand;

};

/** Class to define FaaS pacsltk performance model (static version).
*
*   Class inherited from FaasPE. This class is very similar to FaasPacsltkPE, 
*   but it computes the demand time once and for all, at construction time. 
*   Thus, when the demand time is requested, the stored value is returned, 
*   without the need of recoumputing the same value. Obviously, this model 
*   can be used only under the assumption that the workload does not vary, 
*   which is satisfied for design time problems.
*/
class FaasFixedPE: public FaasPE
{
  public:

    /** FaasFixedPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demandWarm_ Demand time when there is an active server 
    *                      available on the platform
    *   \param demandCold_ Demand time when all the servers on the platform 
    *                      are down and a new one needs to be activated
    *   \param idle_time_before_kill How long does the platform keep the 
    *                                servers up after being idle
    *   \param part_lambda Load factor of the Partition
    */
    FaasFixedPE(
      const std::string& keyword_,
      bool allows_colocation_,
      TimeType demandWarm_,
      TimeType demandCold_,
      TimeType idle_time_before_kill,
      LoadType part_lambda
    );

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data,
      const SolutionData& solution_data
    ) const override;

    /** demandWarm getter */
    TimeType
    get_demandWarm() const override {return demand; }

    /** demandWarm getter */
    TimeType
    get_demandCold() const override {return demand; }

    /** virtual destructor */
    virtual ~FaasFixedPE() = default;

  protected:

    /** Demand time */
    TimeType demand;
};

/**
*   Class to define FaaS performance model based on aMLLibrary
*
*   Class inherited from FaasPE. 
*/
class FaasaMLLibraryPE: public FaasPE
{
  public:
    /** FaasaMLLibrarykPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    *   \param demandWarm_ Demand time when there is an active server 
    *                      available on the platform
    *   \param demandCold_ Demand time when all the servers on the platform 
    *                      are down and a new one needs to be activated
    */
    FaasaMLLibraryPE(
      const std::string& keyword_,
      bool allows_colocation_,
      TimeType demandWarm_,
      TimeType demandCold_,
      const std::string& regressor_file_
    ):
      FaasPE(keyword_, allows_colocation_, demandWarm_, demandCold_),
      regressor_file(regressor_file_),
      predictor(aMLLibrary::Instance())
    {}

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const override;

    virtual ~FaasaMLLibraryPE() = default;

  private:

    /** Regressor file used by aMLLibrary for predictions */
    const std::string regressor_file;

    /** Instance of the singleton class aMLLibrary used to compute the 
    *   demand time through the use of the Python aMLLibrary */
    aMLLibrary& predictor;

};

/**
*   Class to define performance models based on aMLLibrary using cores as the
*   main feature
*/
class CoreBasedaMLLibraryPE: public BasePerformanceModel
{
  public:
    /** CoreBasedaMLLibraryPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    */
    CoreBasedaMLLibraryPE(
      const std::string& keyword_,
      bool allows_colocation_,
      const std::string& regressor_file_
    ):
      BasePerformanceModel(keyword_, allows_colocation_),
      regressor_file(regressor_file_),
      predictor(aMLLibrary::Instance())
    {}

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const override;

    virtual ~CoreBasedaMLLibraryPE() = default;

  private:

    /** Regressor file used by aMLLibrary for predictions */
    const std::string regressor_file;

    /** Instance of the singleton class aMLLibrary used to compute the 
    *   demand time through the use of the Python aMLLibrary */
    aMLLibrary& predictor;

};

/**
*   Class to define performance models based on aMLLibrary using lambda as the
*   main feature
*/
class LambdaBasedaMLLibraryPE: public BasePerformanceModel
{
  public:
    /** CoreBasedaMLLibraryPE constructor.
    *
    *   \param keyword_ Keyword (name) to identify the model to use
    *   \param allows_colocation_ true if colocation is allowed
    */
    LambdaBasedaMLLibraryPE(
      const std::string& keyword_,
      bool allows_colocation_,
      const std::string& regressor_file_
    ):
      BasePerformanceModel(keyword_, allows_colocation_),
      regressor_file(regressor_file_),
      predictor(aMLLibrary::Instance())
    {}

    /** Abstract method of BasePerformanceModel overridden. */
    virtual
    double predict(
      size_t comp_idx, size_t part_idx, ResourceType res_type, size_t res_idx,
      const SystemData& system_data, const SolutionData& solution_data
    ) const override;

    virtual ~LambdaBasedaMLLibraryPE() = default;

  private:

    /** Boolean variable to determine whether the average job execution time 
     * can be used to avoid the pipeline effect */
    const bool meanTime_usage_supported = true;

    /** Regressor file used by aMLLibrary for predictions */
    const std::string regressor_file;

    /** Instance of the singleton class aMLLibrary used to compute the 
    *   demand time through the use of the Python aMLLibrary */
    aMLLibrary& predictor;

};

} //namespace Space4AI

#endif /* PERFORMANCEMODELS_HPP_ */
