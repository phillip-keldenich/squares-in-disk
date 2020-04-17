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
#include <doctest/doctest_fixed.hpp>
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

namespace {
    const auto ite = if_then_else(args::x0 <= args::x1, args::x2, args::x3);
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] if_then_else test", IT, IFloat, IDouble, IRational) {
    IT i1(1), i2(2), i12(1, 2), i13(1, 3), mi(0), pi(0), ii(0);
    mi.set_lb(-infinity);
    pi.set_ub(infinity);
    ii.set_lb(-infinity);
    ii.set_ub(infinity);
    
    REQUIRE_SAME(ite(i1, i2, i1, i2), i1);
    REQUIRE_SAME(ite(i1, i12, i2, i1), i2);
    REQUIRE_SAME(ite(i12, i1, i1, i2), i12);
    REQUIRE_SAME(ite(i12, i13, i1, i12), i12);
    REQUIRE_SAME(ite(i12, i13, i12, i13), i13);
    REQUIRE_SAME(ite(i1, mi, i1, i2), i2);
    REQUIRE_SAME(ite(i1, pi, i2, i1), i12);
    REQUIRE_SAME(ite(i1, pi, mi, pi), ii);
}
