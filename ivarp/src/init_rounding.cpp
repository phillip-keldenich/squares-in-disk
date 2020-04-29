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
// Created by Phillip Keldenich on 2019-10-04.
//

#include "ivarp/rounding.hpp"
#include "ivarp/number.hpp"
#include <mpfr.h>
#include <boost/predef.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

/**
 * @file init_rounding.cpp
 * Initialize the rounding mode at load time, and clean up MPFR caches on exit.
 */

/**
 * Fix interaction between CUDA compiler and g++ C++11 ABI by explicit instantiation of stringstream and ostringstream.
 */
#if BOOST_COMP_GNUC
template class std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>;
template class std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char>>;
#endif

namespace ivarp {
namespace {

/**
 * Attempt to set the rounding mode at shared library load time.
 * On Linux/MacOS, this is sufficient.
 * However, on Windows, apparently, this happens in another thread during DLL loading or
 * something else changes the rounding mode back before the program starts.
 */
static volatile ivarp::SetRoundDown set_round_on_load; // NOLINT

/**
 * @brief Verify that the rounding mode is set at shared object load time.
 * Also verifies that flush-to-zero and denormals-are-zero modes are both turned off.
 */
struct TestRoundOnLoad {
	TestRoundOnLoad() {
		using ivarp::opacify;

		double ddn = opacify(opacify(1.1) * opacify(10.1));
		double dup = -opacify(opacify(-1.1) * opacify(10.1));
		if (ddn >= dup) {
			throw std::runtime_error("Could not verify rounding mode behaves correctly for doubles!"); // LCOV_EXCL_LINE
		}

		float fdn = opacify(opacify(1.1f) * opacify(10.1f));
		float fup = -opacify(opacify(-1.1f) * opacify(10.1f));
		if (fdn >= fup) {
			throw std::runtime_error("Could not verify rounding mode behaves correctly for floats!"); // LCOV_EXCL_LINE
		}

		static_assert(std::numeric_limits<float>::has_denorm != std::denorm_absent, "The platform we are compiling for does not look like x86!");
		static_assert(std::numeric_limits<double>::has_denorm != std::denorm_absent, "The platform we are compiling for does not look like x86!");
		volatile float fden = std::numeric_limits<float>::denorm_min();
		volatile float fmn = std::numeric_limits<float>::min();
		volatile double dden = std::numeric_limits<double>::denorm_min();
		volatile double dmn = std::numeric_limits<double>::min();

		if(fden <= 0.0f || fden + fden <= 0.0f || fden != fden || dden <= 0.0 || dden + dden <= 0.0 || dden != dden || fmn / 2.0f <= fden || dmn / 2.0 <= dden) {
			throw std::runtime_error("Subnormal floating point values are either flushed to zero or treated as zero!");
		}

		if(fden == fmn || dden == dmn) {
			throw std::runtime_error("Subnormal values do not seem to exist!");
		}
	}
};

/**
 * @brief Object to actually run the load-time test.
 */
static volatile TestRoundOnLoad tester; // NOLINT

/**
 * @brief Clear MPFR-internal caches during shutdown.
 */
struct MPFRClearCaches {
    ~MPFRClearCaches() {
        mpfr_free_cache();
        mpfr_free_cache2(MPFR_FREE_GLOBAL_CACHE);
        mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
    }
};

/**
 * @brief Object to actually clear the caches during shutdown.
 */
static volatile MPFRClearCaches clear_caches_on_exit; // NOLINT

}
}
