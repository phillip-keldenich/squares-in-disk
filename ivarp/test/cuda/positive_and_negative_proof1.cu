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
// Created by Phillip Keldenich on 16.04.2020.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/run_prover.hpp"
#include "ivarp/atomic_float.hpp"

namespace {
    using namespace ivarp;
    const auto x = args::x0; // x is the first variable
    const auto y = args::x1; // y is the second variable
    const auto f = fixed_pow<3>(x) - 2 * square(y) - 3 * x + sqrt(y); // the function f

    const auto constrs_sat = constraint_system(
        variable(x, "x", -2_Z, 2_Z), variable(y, "y", 0_Z, 2_Z),
        f >= 2.37_X, y >= x
    );
    const auto constrs_unsat = constraint_system(
        variable(x, "x", -2_Z, 2_Z), variable(y, "y", 0_Z, 2_Z),
        f >= 2.38_X, y >= x
    );

    using VarSplit = U64Pack<dynamic_subdivision(8,4), 64>;
    using Context = DefaultContextWithNumberType<IDouble>;

    struct Joiner {
        std::atomic<double> max_x{-std::numeric_limits<double>::infinity()};
        std::atomic<double> min_x{std::numeric_limits<double>::infinity()};
        std::atomic<double> max_y{-std::numeric_limits<double>::infinity()};
        std::atomic<double> min_y{std::numeric_limits<double>::infinity()};
    };

    struct JoinHandler {
        JoinHandler(Joiner* joiner) noexcept : joiner(joiner) {}

        // this code is run on all remaining critical hypercuboids (potentially by multiple threads)
        template<typename CTX, typename Crit> void operator()(const CTX&, const Crit& critical) const {
            atomic_max(&joiner->max_x, critical[0].ub());
            atomic_max(&joiner->max_y, critical[1].ub());
            atomic_min(&joiner->min_x, critical[0].lb());
            atomic_min(&joiner->min_y, critical[1].lb());
        }

        Joiner* joiner;
    };

    TEST_CASE("[ivarp][prover][cuda] Negative proof test case 1") {
        Joiner joiner;
        JoinHandler handler(&joiner);

        ProverSettings settings;
        settings.generation_count = 2;
        settings.max_iterations_per_node = 1;
		ProofInformation info;

        // create input and run prover
        const auto input = prover_input<Context, VarSplit>(constrs_sat);
        REQUIRE(!run_prover(input, handler, &info, settings));

        // check that the ranges for x and y are not empty.
        REQUIRE(joiner.min_x < joiner.max_x);
        REQUIRE(joiner.min_y < joiner.max_y);
		REQUIRE(info.num_cuboids > 0);
		REQUIRE(info.num_leaf_cuboids > 0);
		REQUIRE(info.num_critical_cuboids > 0);

        // check that the mid point actually violates the <= 2.37 condition.
        double midx = 0.5 * (joiner.max_x.load() + joiner.min_x.load());
        double midy = 0.5 * (joiner.max_y.load() + joiner.min_y.load());
        REQUIRE(f(midx, midy) >= 2.37);
    }

    TEST_CASE("[ivarp][prover][cuda] Positive proof test case 1") {
        Joiner joiner;
        JoinHandler handler(&joiner);

        ProverSettings settings;
		ProofInformation info;
        settings.generation_count = 7;
        settings.max_iterations_per_node = 1;

        // create input and run prover
        const auto input = prover_input<Context, VarSplit>(constrs_unsat);
        REQUIRE(run_prover(input, handler, &info, settings));
    }
}
