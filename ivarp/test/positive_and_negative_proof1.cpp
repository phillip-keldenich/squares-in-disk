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
// Created by Phillip Keldenich on 18.11.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/prover.hpp"
#include "ivarp/atomic_float.hpp"

using namespace ivarp;

namespace {
    // Prove a bound on x^3 - 2*y^2 - 3*x + sqrt(y) with x in [-2,2] and y in [0,2] with y >= x
    // see https://www.wolframalpha.com/input/?i=%28x%5E3+-+2y%5E2+-+3x+%2B+sqrt%28y%29+%3E%3D+2.37%29+for+x+from+-2+to+2+and+y+from+0+to+2
    // for an incorrect wolframalpha plot :-) that makes it look like constrs_sat is actually unsatisfiable (it isn't).
    const auto x = args::x0; // x is the first variable
    const auto y = args::x1; // y is the second variable
    const auto vars = variables(variable<-8>(x, "x", -2, 2), variable<64>(y, "y", 0, 2)); // variable x in [-2,2] is dynamically split with 8 initial subintervals; y is split into 64 subintervals
    const auto f = fixed_pow<3>(x) - 2 * square(y) - 3 * x + sqrt(y); // the function f
    const auto constrs_sat = predicate_and_constraints(f <= constant(2.37), y >= x);   // first argument must be true for all values that
    const auto constrs_unsat = predicate_and_constraints(f <= constant(2.38), y >= x); // conform to the constraints (other arguments)

    const auto prover_success = prover(vars, constrs_unsat); // a prover (that should succeed) for constrs_unsat
    const auto prover_failure = prover(vars, constrs_sat);   // a prover (that should fail) for constrs_sat
}

struct Proof1Settings : DefaultProverSettings { // default settings except for:
    static constexpr std::size_t max_dynamic_split_depth = 7; // the maximum number of times x is split (into 4 subintervals) is 7 instead of the default (12)
};

TEST_CASE("[ivarp][prover] Negative proof test case 1") {
    std::atomic<double> max_x{-std::numeric_limits<double>::infinity()};
    std::atomic<double> min_x{std::numeric_limits<double>::infinity()};
    std::atomic<double> max_y{-std::numeric_limits<double>::infinity()};
    std::atomic<double> min_y{std::numeric_limits<double>::infinity()};

    const auto reporter = [&] (const auto& /*ctx*/, const auto& critical) {
		// this code is run on all remaining critical hypercuboids (by multiple threads)
        atomic_max(&max_x, critical[0].ub());
        atomic_max(&max_y, critical[1].ub());
        atomic_min(&min_x, critical[0].lb());
        atomic_min(&min_y, critical[1].lb());
    };

	// actually run the prover; this should fail.
    REQUIRE(!prover_failure.run<Proof1Settings>(reporter));

	// check that the ranges for x and y are not empty.
    REQUIRE(min_x < max_x);
    REQUIRE(min_y < max_y);

	// check that the mid point actually violates the <= 2.37 condition and WolframAlpha's plot is a dirty liar.
    double midx = 0.5 * (max_x.load() + min_x.load());
    double midy = 0.5 * (max_y.load() + min_y.load());
    REQUIRE(f(midx, midy) >= 2.37);
}

TEST_CASE("[ivarp][prover] Positive proof test case 1") {
    const auto reporter = [] (const auto& /*ctx*/, const auto& critical) {
        std::ostringstream text;
        text << "ERROR: Critical reported x = " << critical[0] << ", y = " << critical[1] << std::endl;
        std::cout << text.str();
        REQUIRE(false);
    };
    REQUIRE(prover_success.run<Proof1Settings>(reporter));
}
