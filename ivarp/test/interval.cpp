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
// Created by Phillip Keldenich on 10.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"

using namespace ivarp;

static_assert(std::is_trivially_copy_constructible<IFloat>::value, "Error!");
static_assert(std::is_trivially_copy_assignable<IFloat>::value, "Error!");
static_assert(std::is_trivially_copy_constructible<IDouble>::value, "Error!");
static_assert(std::is_trivially_copy_assignable<IDouble>::value, "Error!");

TEST_CASE_TEMPLATE("[ivarp][number] Interval default constructors", IT, IRational) {
    IT dfl;
    REQUIRE(lb(dfl) == dfl.lb());
    REQUIRE(ub(dfl) == dfl.ub());
    REQUIRE(lb(dfl) == 0);
    REQUIRE(ub(dfl) == 0);
    REQUIRE(dfl.contains(0));
    REQUIRE(dfl.same(IT{0,0,false}));
    REQUIRE(!dfl.same(IT{0,0,true}));
    REQUIRE(dfl.same(IT{0}));
}
