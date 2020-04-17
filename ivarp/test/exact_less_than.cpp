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
// Created by Phillip Keldenich on 28.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"

using namespace ivarp;

TEST_CASE_TEMPLATE("[ivarp][number] exact_less_than", FT, float, double) {
    REQUIRE(exact_less_than(FT(0), FT(1)));
    REQUIRE(!exact_less_than(FT(1), FT(0)));
    REQUIRE(exact_less_than(FT(0), 1));
    REQUIRE(!exact_less_than(1, FT(0)));

    std::uint64_t too_large1 = std::uint64_t(1u) << 55u;
    std::uint64_t too_large2 = too_large1 + 1u;
    int okay_d = 87654321;
    double max_exact = 18014398509481984.0;
    double max_exactp = 18014398509481985.0;
    REQUIRE(max_exact == max_exactp);
    double max_exactm1 = max_exact - 1;
    std::uint64_t okay_d64(okay_d);
    std::int64_t okay_i64(okay_d);

    REQUIRE((double)(float)okay_d != (double)okay_d);
    REQUIRE(exact_less_than(float(okay_d), okay_d) == ((double)(float)okay_d < (double)okay_d));
    REQUIRE(exact_less_than(float(okay_d), okay_d64) == ((double)(float)okay_d < (double)okay_d));
    REQUIRE(exact_less_than(float(okay_d), okay_i64) == ((double)(float)okay_d < (double)okay_d));
    REQUIRE(exact_less_than(-float(okay_d), -okay_i64) == !((double)(float)okay_d < (double)okay_d));

    REQUIRE(exact_less_than(FT(too_large1), too_large2));
    REQUIRE(!exact_less_than(FT(too_large1), too_large1));
    REQUIRE(!exact_less_than(FT(too_large2), too_large1));
    REQUIRE(exact_less_than(FT(too_large2), too_large2));
    REQUIRE(exact_less_than(FT(55), 750));
    REQUIRE(exact_less_than(FT(-1333), 1333));
    REQUIRE(!exact_less_than(FT(-1333), -1333));
    REQUIRE(exact_less_than(-std::numeric_limits<FT>::max(), -std::numeric_limits<std::int64_t>::max()));
    REQUIRE(exact_less_than(-std::numeric_limits<FT>::infinity(), -std::numeric_limits<std::int64_t>::max()));
    REQUIRE(exact_less_than(std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<FT>::max()));
    REQUIRE(exact_less_than(std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<FT>::infinity()));
    REQUIRE(!exact_less_than(FT(std::numeric_limits<std::uint64_t>::max()/2+1), // NOLINT
                             std::numeric_limits<std::uint64_t>::max()/2+1));
    REQUIRE(exact_less_than(max_exactm1, max_exact));
    REQUIRE(exact_less_than(-max_exact, -max_exactm1));
    REQUIRE(exact_less_than(max_exact, too_large2));
    REQUIRE(exact_less_than(max_exact, (1ull<<54u) + 1));
    REQUIRE(exact_less_than(-(1ll<<54u) - 1, -max_exact)); // NOLINT
    REQUIRE(!exact_less_than((1ll<<54u) + 1, max_exact)); // NOLINT
    REQUIRE(exact_less_than(534.0, 1ull<<56u));
    REQUIRE(exact_less_than(-1ll * (1ll<<56u), -999.9)); // NOLINT
}
