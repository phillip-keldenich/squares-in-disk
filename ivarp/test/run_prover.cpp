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
// Created by Phillip Keldenich on 10.02.20.
//

#include "ivarp/run_prover.hpp"
#include "test_util.hpp"

using namespace ivarp;

namespace run_prover_test {
namespace {
    // Prove a bound on x^3 - 2*y^2 - 3*x + sqrt(y) with x in [-2,2] and y in [0,2] with y >= x
    // see https://www.wolframalpha.com/input/?i=%28x%5E3+-+2y%5E2+-+3x+%2B+sqrt%28y%29+%3E%3D+2.37%29+for+x+from+-2+to+2+and+y+from+0+to+2
    // for an incorrect wolframalpha plot :-) that makes it look like system_sat is actually unsatisfiable (it isn't).
    const auto x = args::x0;
    const auto y = args::x1;
    const auto cube = fixed_pow<3>(x);
    const auto f = cube(x) - 2_Z * square(y) - 3_Z * x + sqrt(y);
    const auto system_sat = constraint_system(variable(x, "x", -2_Z, 2_Z), variable(y, "y", 0, 2), y >= x, f >= 2.37_X);
    const auto system_unsat = constraint_system(variable(x, "x", -2_Z, 2_Z), variable(y, "y", 0, 2), y >= x, f >= 2.38_X);
}

    TEST_CASE_TEMPLATE("[ivarp][run_prover] Negative proof result", NT, IFloat, IDouble) {
        using Context = DefaultContextWithNumberType<NT>;
        using VariableSplitting = U64Pack<dynamic_subdivision(64,2), 32>;
        const auto lower_bound = convert_number<NT>(2.37_X);
        const auto input = prover_input<Context,VariableSplitting>(system_sat);
        //FunctionPrinter fprinter{ PrintOptions{}, &system_sat };
        //print_prover_input(std::cout, input, fprinter);
        REQUIRE(!input.initial_runtime_bounds[0].possibly_undefined());
        REQUIRE(!input.initial_runtime_bounds[1].possibly_undefined());
        std::atomic<std::size_t> oc_called{0};
        bool one_def = false;
        auto on_crit = [&] (const auto& /*ctx*/, const NT* values) {
            ++oc_called;
            NT result = f.template array_evaluate<Context>(values);
            REQUIRE(possibly(result >= lower_bound));
            one_def |= definitely(result >= lower_bound);
        };
        ProofInformation info;
        ProverSettings settings;
        settings.generation_count = 11;
        settings.max_iterations_per_node = 1;
        REQUIRE(!run_prover(input, on_crit, &info, settings));
        REQUIRE(one_def);
        REQUIRE(oc_called > 0);
        REQUIRE(info.num_cuboids > 64 * 32);
        REQUIRE(info.num_leaf_cuboids > info.num_cuboids / 2);
        REQUIRE(info.num_critical_cuboids == oc_called.load());
        REQUIRE(info.num_cuboids > info.num_leaf_cuboids);
        REQUIRE(info.num_leaf_cuboids > info.num_critical_cuboids);
    }

    TEST_CASE_TEMPLATE("[ivarp][run_prover] Positive proof result", NT, IFloat, IDouble) {
        using Context = DefaultContextWithNumberType<NT>;
        using VariableSplitting = U64Pack<dynamic_subdivision(64,2), 32>;
        const auto input = prover_input<Context,VariableSplitting>(system_unsat);
        REQUIRE(!input.initial_runtime_bounds[0].possibly_undefined());
        REQUIRE(!input.initial_runtime_bounds[1].possibly_undefined());
        auto on_crit = [&] (const auto& /*ctx*/, const NT*) {
            REQUIRE(false);
        };
        ProofInformation info;
        ProverSettings settings;
        settings.generation_count = 11;
        settings.max_iterations_per_node = 1;
        REQUIRE(run_prover(input, on_crit, &info, settings));
        REQUIRE(info.num_cuboids > 64 * 32);
        REQUIRE(info.num_leaf_cuboids > info.num_cuboids / 2);
        REQUIRE(info.num_cuboids > info.num_leaf_cuboids);
        REQUIRE(info.num_critical_cuboids == 0);
    }
}
