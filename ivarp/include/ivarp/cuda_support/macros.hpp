// The code is open source under the MIT license.
// Copyright 2019-2020, Phillip Keldenich, TU Braunschweig, Algorithms Group
// https://ibr.cs.tu-bs.de/alg
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Created by Phillip Keldenich on 19.01.20.
//

/**
 * @file macros.hpp
 * Contains macros, used mainly for optionally supporting CUDA.
 */
#pragma once

#include <type_traits>
#include <boost/predef.h>

/**
 * \def IVARP_PREVENT_MACRO
 * \brief A macro that expands to nothing.
 *
 * Useful to prevent macro substitution, e.g., min
 * in the first line of the following example:
 * \code
 *   std::min(a,b)
 *   (std::min)(a,b)
 *   std::min IVARP_PREVENT_MACRO (a,b)
 * \endcode
 */
#define IVARP_PREVENT_MACRO

/**
 * \def IVARP_PRAGMA_KEYWORD
 * \brief Expands to the (compiler, not preprocessor) pragma
 * keyword, which is compiler-dependent.
 */
#if defined(_MSC_VER) && !defined(__clang__) // BOOST_COMP_MSVC does not work with nvcc running on MSVC
#define IVARP_PRAGMA_KEYWORD __pragma
#else
#define IVARP_PRAGMA_KEYWORD _Pragma
#endif

#ifndef __CUDACC__

/**
 * \def IVARP_HD
 * \brief A macro that is used to mark functions
 * as device & host functions (i.e., functions that may
 * be called in both CPU host and CUDA device code.)
 */
#define IVARP_HD

/**
 * \def IVARP_H
 * \brief A macro that is used to mark functions
 * as host-only functions (i.e., functions that may
 * not be called in CUDA device code.)
 */
#define IVARP_H

/**
 * \def IVARP_D
 * \brief A macro that is used to mark functions
 * as device-only functions (i.e., functions that may
 * only be called in CUDA device code.)
 */
#define IVARP_D

/**
 * \def \_\_global\_\_
 * \brief Provide an empty definition for \_\_global\_\_
 * if CUDA is not available.
 */
#ifndef __global__
#define __global__
#endif

/**
 * \def IVARP_SUPPRESS_HD
 * \brief Suppress CUDA host/device-function related warnings.
 * Note that this does not magically allow calling host functions from
 * device code.
 */
#define IVARP_SUPPRESS_HD

#else

#define IVARP_HD __host__ __device__
#define IVARP_H __host__
#define IVARP_D __device__

#define IVARP_SUPPRESS_HD IVARP_PRAGMA_KEYWORD ("hd_warning_disable")

#endif

/**
 * \def IVARP_DEFAULT_CM(T)
 * \brief For a class \a T, define defaulted
 * move and copy constructors and operators,
 * suppressing all host/device function warnings.
 */
#define IVARP_DEFAULT_CM(T)\
        IVARP_SUPPRESS_HD T(const T&) = default;\
        IVARP_SUPPRESS_HD T(T&&) = default;\
        IVARP_SUPPRESS_HD T& operator=(const T&) = default;\
        IVARP_SUPPRESS_HD T& operator=(T&&) = default

/**
 * \def IVARP_ENABLE_FOR_CUDA_NT(NumberType)
 * \brief For a number type \a NumberType, expands to a
 * sequence of template parameters that can be added to a function template
 * in order to only enable it by SFINAE if the given \a NumberType supports CUDA.
 */
#define IVARP_ENABLE_FOR_CUDA_NT(NumberType)\
    typename NT = NumberType, std::enable_if_t<AllowsCUDA<NT>::value, int> = 0

 /**
 * \def IVARP_DISABLE_FOR_CUDA_NT(NumberType)
 * \brief For a number type \a NumberType, expands to a
 * sequence of template parameters that can be added to a function template
 * in order to disable it by SFINAE if the given \a NumberType supports CUDA.
 */
#define IVARP_DISABLE_FOR_CUDA_NT(NumberType)\
    typename NT = NumberType, std::enable_if_t<!AllowsCUDA<NT>::value, int> = 0

/// Create two overloaded definitions of essentially the same function template;
/// the only difference is whether CUDA is supported. CUDA support depends on the given NumberType.
/// This essentially allows to have a function that is marked \_\_host\_\_ \_\_device\_\_ if supported and \_\_host\_\_ otherwise.
#define IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType, ...)\
    template<IVARP_ENABLE_FOR_CUDA_NT(NumberType)> IVARP_HD __VA_ARGS__\
    template<IVARP_DISABLE_FOR_CUDA_NT(NumberType)> IVARP_H __VA_ARGS__

/**
 * \def IVARP_TEMPLATE_PARAMS(...)
 * \brief Expands to all given arguments;
 * used to group a set of template parameters
 * given as argument to a macro without needing
 * the macro varargs for the template parameters.
 */
#define IVARP_TEMPLATE_PARAMS(...) __VA_ARGS__

/// Create two overloaded definitions of essentially the same function template;
/// the only difference is whether CUDA is supported. CUDA support depends on the given NumberType.
/// This essentially allows to have a function template that is marked \_\_host\_\_ \_\_device\_\_ if supported and \_\_host\_\_ otherwise.
/// Compared to #IVARP_HD_OVERLOAD_ON_CUDA_NT , it is possible to give additional template parameters
/// for the function template.
#define IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(TemplateParams, NumberType, ...)\
    template<TemplateParams, IVARP_ENABLE_FOR_CUDA_NT(NumberType)> IVARP_HD __VA_ARGS__\
    template<TemplateParams, IVARP_DISABLE_FOR_CUDA_NT(NumberType)> IVARP_H __VA_ARGS__
