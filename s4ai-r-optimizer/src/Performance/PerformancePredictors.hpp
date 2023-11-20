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
* \file PerformancePredictors.hpp.in
*
* \brief Defines the classes that provide the interface between the Performance
*        Models (C++) and the (Python) functions.
*
* C++/Python interface is provided by a Web Server implemented with the Gunicorn WSGI.
* The communication between the library and the server is provided by the CPR library.
*
* \author Randeep Singh
*/

#ifndef PERFORMANCEPREDICTORS_HPP_
#define PERFORMANCEPREDICTORS_HPP_

#include "src/TypeTraits.hpp"

namespace Space4AI
{

/**
 * BasePredictor
*/
class BasePredictor
{
  protected:

    /** Default constructor */
    BasePredictor() = default;

  public:

    /** Method to compute the demand time */
    virtual TimeType predict(const nlohmann::json& features) const = 0;

    BasePredictor(const BasePredictor&) = delete;
    BasePredictor& operator=(const BasePredictor&) = delete;

    virtual ~BasePredictor() = default;
};

/**
 * Pacsltk
*/
class Pacsltk: public BasePredictor
{
  protected:

    /** Default constructor */
    Pacsltk() = default;

  public:

    /** Instance getter */
    static
    Pacsltk&
    Instance()
    {
      // We use the Meyer's trick to instantiate the factory as Singleton
      static Pacsltk single_pacsltk;
      return single_pacsltk;
    }

    /** Method to compute the demand time */
    TimeType predict(const nlohmann::json& features) const override;

    Pacsltk(const Pacsltk&) = delete;
    Pacsltk& operator=(const Pacsltk&) = delete;

    virtual ~Pacsltk() = default;
};

/**
 * aMLLibrary
*/
class aMLLibrary: BasePredictor
{
  protected:

    /** Default constructor */
    aMLLibrary() = default;

  public:

    /** Instance getter */
    static
    aMLLibrary&
    Instance()
    {
      // We use the Meyer's trick to instantiate the factory as Singleton
      static aMLLibrary single_amllibrary;
      return single_amllibrary;
    }

    /** Method to compute the demand time */
    TimeType predict(const nlohmann::json& features) const override;

    aMLLibrary(const aMLLibrary&) = delete;
    aMLLibrary& operator=(const aMLLibrary&) = delete;

    virtual ~aMLLibrary() = default;
};

} //namespace Space4AI

#endif /* PERFORMANCEPREDICTORS_HPP_ */
