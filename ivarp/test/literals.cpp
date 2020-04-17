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
// Created by Phillip Keldenich on 17.12.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"

using namespace ivarp;

TEST_CASE("[ivarp][number] Decimal literals") {
    const auto test1 = 1.000000000010000000000000000000000000000000000001000000000000000000000000_X;
    using T1 = std::decay_t<decltype(test1)>;
    static_assert(T1::lb == 1'000000, "Wrong lower bound!");
    static_assert(T1::ub == 1'000001, "Wrong upper bound!");
    Rational ex1("1000000000010000000000000000000000000000000000001/"
                 "1000000000000000000000000000000000000000000000000");
    REQUIRE(ex1 == test1.value);

    const auto test2 = 55.000000000000000000000000000000000000000000000000000000000000000000_X;
    using T2 = std::decay_t<decltype(test2)>;
    static_assert(T2::lb == 55'000000 && T2::ub == 55'000000, "Wrong bounds!");
    REQUIRE(test2.value == 55);
}
