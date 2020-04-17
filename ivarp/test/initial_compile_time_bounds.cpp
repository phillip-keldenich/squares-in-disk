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
// Created by Phillip Keldenich on 14.01.20.
//

#include "ivarp/prover.hpp"
#include "test_util.hpp"

using namespace ivarp;
using namespace ivarp::args;

namespace {
    const auto vars1 = variables(variable<-8>(x0, "x0", 2_Z, 11_Z), variable<8>(x1, "x1", 0.555_X, 2_Z * x0), variable<8>(x2, "x2", -1_Z, x0 + x1));
    using V1IB = decltype(vars1)::InitialCompileTimeBounds;
    using V1IB_0 = V1IB::At<0>;
    using V1IB_1 = V1IB::At<1>;
    using V1IB_2 = V1IB::At<2>;

    static_assert(V1IB_0::lb == fixed_point_bounds::int_to_fp(2), "Wrong lower bound for x0!");
    static_assert(V1IB_0::ub == fixed_point_bounds::int_to_fp(11), "Wrong upper bound for x0!");
    static_assert(V1IB_1::lb == 555 * fixed_point_bounds::denom() / 1000, "Wrong lower bound for x1!");
    static_assert(V1IB_1::ub == fixed_point_bounds::int_to_fp(22), "Wrong upper bound for x1!");
    static_assert(V1IB_2::lb == fixed_point_bounds::int_to_fp(-1), "Wrong lower bound for x2!");
    static_assert(V1IB_2::ub == fixed_point_bounds::int_to_fp(33), "Wrong upper bound for x2!");
}
