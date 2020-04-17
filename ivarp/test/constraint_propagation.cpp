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
// Created by Phillip Keldenich on 15.11.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/prover.hpp"
#include "test_util.hpp"

using namespace ivarp;

TEST_CASE_TEMPLATE("[ivarp][constraint propagation] Simple bound propagation test", IT, IFloat, IDouble, IRational) {
    const auto x = args::x0;
    const auto y = args::x1;

    const auto vars = variables(variable<-4>(x, "x", 0, 1), variable<256>(y, "y", 0, 1));
    using Ctx = DefaultContextWithNumberType<IT>;
    using Ary = Array<IT, 2>;

    Ary args;
    args[0] = vars[0_i].compute_bounds(Ctx{}, args);
    REQUIRE_SAME(args[0], IT(0,1));
    REQUIRE(!args[0].empty());
    args[1] = vars[1_i].compute_bounds(Ctx{}, args);
    REQUIRE_SAME(args[1], IT(0,1));
    REQUIRE(!args[1].empty());

    const auto constrs = constraints(y <= 2*x, x >= y/2);
    const auto cprop = constraint_propagation(&vars, &constrs);
    const Splitter<IT> xsplit(args[0], 4);

    using Constraints = std::decay_t<decltype(constrs)>;
    using Constraint1 = Constraints::ByIndex<0>;
    using Constraint2 = Constraints::ByIndex<1>;
    static_assert(!std::is_same<Constraint1,Constraint2>::value, "The constraints are different!");
    static_assert(CanReformulateIntoBound<Constraint1, MathArgFromIndex<1>>::value, "Should be able to reformulate!");
    static_assert(CanReformulateIntoBound<Constraint2, MathArgFromIndex<0>>::value, "Should be able to reformulate!");
    static_assert(
        std::is_same<typename Constraints::template RewritableVariables<Constraint1, 2>, IndexPack<1>>::value,
        "Should be rewritable!"
    );
    static_assert(
        std::is_same<typename Constraints::template RewritableVariables<Constraint1, 2, 1>, IndexPack<>>::value,
        "Should be rewritable!"
    );
    static_assert(
        std::is_same<typename Constraints::template RewritableVariables<Constraint1, 2, 0>, IndexPack<1>>::value,
        "Should be rewritable!"
    );
    static_assert(
        std::is_same<typename Constraints::template RewritableVariables<Constraint2, 2>, IndexPack<0>>::value,
        "Should be rewritable!"
    );
    static_assert(
        std::is_same<typename Constraints::template RewritableVariables<Constraint2, 2, 0>, IndexPack<>>::value,
        "Should be rewritable!"
    );


    args[0] = xsplit.subrange(0);
    REQUIRE_SAME(args[0], IT(0,0.25f));
    REQUIRE(!cprop.template compute_new_var_bounds<1, Ctx>(args).empty);
    REQUIRE_SAME(args[1], IT(0,0.5f));
    args[1] = IT(0.125f, 0.25f);
    REQUIRE(!cprop.template dynamic_post_split_var<1, 2, Ctx>(args).empty);
    REQUIRE_SAME(args[0], IT(0.0625f, 0.25f));
    REQUIRE_SAME(args[1], IT(0.125f, 0.25f));
}

TEST_CASE_TEMPLATE("[ivarp][prover] Simple initial queue test", IT, IFloat, IDouble) {
    const auto x = args::x0;
    const auto y = args::x1;
    const auto vars = variables(variable<-16>(x, "x", 0, 1), variable<-8>(y, "y", 0, 1));
    const auto constrs = constraints(y <= 2*x, x >= y/2);
    const auto prf = prover(vars, constrs);
    using ProverSettings = DefaultProverSettingsWithNumberType<IT>;
	const auto initial_queue_content = prf.template generate_dynamic_phase_queue<ProverSettings>();
    REQUIRE(initial_queue_content.size() == 128);
    Splitter<IT> xsplit(IT(0,1), 16);
    std::size_t idx = 0;
    for(int xi = 0; xi < 16; ++xi) {
        IT xsub = xsplit.subrange(xi);
        auto ymax = 2*xsub.ub();
        if(ymax > 1) { ymax = 1; }
        Splitter<IT> ysplit(IT(0, ymax), 8);
        for(int yi = 0; yi < 8; ++yi, ++idx) {
            REQUIRE(!initial_queue_content[idx].empty);
            REQUIRE(initial_queue_content[idx].depth == 0);
            IT ivx = initial_queue_content[idx].bounds[0];
            IT ivy = initial_queue_content[idx].bounds[1];
            IT rgy = ysplit.subrange(yi);

            REQUIRE(ivx.lb() == (std::max)(xsub.lb(), ivy.lb() / 2));
            REQUIRE(ivx.ub() == xsub.ub());
            REQUIRE(ivy.lb() == rgy.lb());
            REQUIRE(ivy.ub() == rgy.ub());
        }
    }
}
