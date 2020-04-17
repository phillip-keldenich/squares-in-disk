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
// Created by Phillip Keldenich on 17.11.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/prover.hpp"

using namespace ivarp;

namespace {
    const auto x = args::x0;
    const auto y = args::x1;
    const auto z = args::x2;
    const auto bound_fn = constant(5)/2 - args::x0;

    static_assert(impl::NumChildren<std::decay_t<decltype(bound_fn(x))>>::value == 2, "Wrong number of children!");
    static_assert(impl::NumChildren<std::decay_t<decltype(x <= bound_fn(y))>>::value == 2, "Wrong number of children!");

    const auto vars = variables(variable<-8>(x, "x", 0, 2.4999_X),
                                variable<-8>(y, "y", 0, 2.4999_X),
                                variable<64>(z, "z", 0, 2.4999_X));

    const auto constrs = predicate_and_constraints(x+y+z <= 5,
                                                   x <= bound_fn(y),
                                                   y <= bound_fn(x),
                                                   y <= bound_fn(z),
                                                   z <= bound_fn(y));
    const auto prv = prover(vars, constrs);

    using Variables = std::decay_t<decltype(vars)>;
    static_assert(Variables::initial_dynamic_cuboids == 64, "Wrong initial queue size!");
}

TEST_CASE("[ivarp][prover] Simple positive proof test") {
    auto handler = [] (const auto& /*ctx*/, const auto& critical) {
        std::cerr << "Error: Reported critical: x = " << critical[0] << ", y = " << critical[1]
                  << ", z = " << critical[2] << std::endl;
        REQUIRE(false);
    };

    REQUIRE(prv.run(handler));
}
