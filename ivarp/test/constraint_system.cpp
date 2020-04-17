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
// Created by Phillip Keldenich on 06.02.20.
//

#include "ivarp/constraint_system.hpp"
#include "test_util.hpp"

using namespace ivarp;

namespace {
namespace test1 {
    using fixed_point_bounds::int_to_fp;
    const auto x = args::x0;
    const auto y = args::x1;
    const auto z = args::x2;
    const auto w = args::x3;
    auto cs = constraint_system(
        variable(x, "x", 0_Z, 10_Z), variable(z, "z", -y, y), variable(y, "y", 0_Z, x),
        value(w, x+y*z, "w"), x + y >= 5_Z, z <= w
    );
    using CSType = decltype(cs);
    static_assert(std::is_same<typename CSType::VariableIndices, IndexPack<0,1,2>>::value, "Wrong type!");
    using Bounds = CSType::CTBounds;
    using BX = Bounds::At<0>;
    static_assert(BX::lb == 0 && BX::ub == int_to_fp(10), "Wrong bounds!");
    using BY = Bounds::At<1>;
    static_assert(BY::lb == 0 && BY::ub == int_to_fp(10), "Wrong bounds!");
    using BZ = Bounds::At<2>;
    static_assert(BZ::lb == -int_to_fp(10) && BZ::ub == int_to_fp(10), "Wrong bounds!");
}
}
