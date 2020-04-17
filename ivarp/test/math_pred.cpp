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
// Created by Phillip Keldenich on 24.10.19.
//

#include "ivarp/math_fn.hpp"
#include <doctest/doctest_fixed.hpp>

using namespace ivarp;

auto sine = sin(args::x0);
auto cosine = cos(args::x0);
auto comp = sine(args::x0) <= cosine(args::x0) || !(sine(args::x0) * args::x1 <= args::x1 / 2);
auto comp2 = comp(args::x0 + args::x1, args::x0 - args::x1);

auto always_true = (true && constant(true)) || constant(false) || args::x0 <= 0;

static_assert(NumArgs<decltype(comp)>::value == 2, "Wrong number of arguments!");
static_assert(NumArgs<decltype(sine(args::x0) == cosine(args::x2))>::value == 3, "Wrong number of arguments!");

TEST_CASE_TEMPLATE("[ivarp][math_fn] Simple predicate test", NT, float, double, Rational, IFloat, IDouble, IRational) {
    REQUIRE(definitely(comp(NT(0), NT(5))));
    REQUIRE(definitely(comp(NT(0.1f), NT(5))));
    REQUIRE(definitely(comp(NT(1),  NT(5))));
    REQUIRE(definitely(always_true(NT(1))));
}
