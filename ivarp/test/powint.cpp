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
// Created by Phillip Keldenich on 21.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

static const auto sqr = ivarp::square(args::x0);
static const auto cube = ivarp::fixed_pow<3>(args::x0);
static const auto pow4 = ivarp::fixed_pow<4>(args::x0);

TEST_CASE_TEMPLATE("[ivarp][math_fn] fixed_pow evaluation on floating-point numbers", FT, float, double, Rational) {
    REQUIRE(sqr(FT(-1)) == 1);
    REQUIRE(sqr(FT(1)) == 1);
    REQUIRE(cube(FT(-1)) == -1);
    REQUIRE(cube(FT(1)) == 1);
    REQUIRE(sqr(FT(2)) == 4);
    REQUIRE(sqr(FT(-2)) == 4);
    REQUIRE(cube(FT(2)) == 8);
    REQUIRE(cube(FT(-2)) == -8);
    REQUIRE(pow4(FT(-1)) == 1);
    REQUIRE(pow4(FT(1)) == 1);
    REQUIRE(pow4(FT(2)) == 16);
    REQUIRE(pow4(FT(-2)) == 16);
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] fixed_pow evaluation on intervals with infinities", FT, float, double, Rational) {
    using IV = Interval<FT>;
    IV nonneg{0, infinity};
    IV nonpos{-infinity, 0};
    IV real{-infinity,infinity};
    REQUIRE_SAME(nonneg, sqr(real));
    REQUIRE_SAME(nonneg, sqr(nonpos));
    REQUIRE_SAME(nonneg, sqr(nonneg));
    REQUIRE_SAME(nonneg, pow4(real));
    REQUIRE_SAME(nonneg, pow4(nonpos));
    REQUIRE_SAME(nonneg, pow4(nonneg));
    REQUIRE_SAME(real, cube(real));
    REQUIRE_SAME(nonpos, cube(nonpos));
    REQUIRE_SAME(nonneg, cube(nonneg));
    REQUIRE_SAME(impl::rational_ipow(IRational(-infinity,infinity), 0), IRational(1,1));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Random interval fixed pow", FT, float, double, Rational) {
    IDouble range{-10,10};
    IDouble widths[] = {
        { 0, 0.2 },
        { 0.2, 0.9 },
        {0.9, 1.8 },
        {1.8, 4 }
    };
    random_test_unary_operator<FT>(sqr, range, std::begin(widths), std::end(widths));
    random_test_unary_operator<FT>(cube, range, std::begin(widths), std::end(widths));
    random_test_unary_operator<FT>(pow4, range, std::begin(widths), std::end(widths));
}

TEST_CASE("[ivarp][math_fn] Regression test for cube") {
    IDouble v{-3.7678346042157239, -2.6545802008270055 };
    IRational r{Rational{"-8484409259769797/2251799813685248"},
                Rational{"-5977583201634799/2251799813685248"}};
    REQUIRE(v.lb() <= r.lb());
    REQUIRE(v.ub() >= r.ub());
    REQUIRE(r.same(convert_number<IRational>(v)));

    // perform correctness checks, including a check that the rational interval is tighter
    Rational lower = cube(r.lb()), upper = cube(r.ub());
    REQUIRE(r.lb() * r.lb() * r.lb() == lower);
    REQUIRE(r.ub() * r.ub() * r.ub() == upper);
    IDouble op_v = cube(v);
    IRational op_r = cube(r);
    REQUIRE(lower == op_r.lb());
    REQUIRE(upper == op_r.ub());
    REQUIRE(op_v.contains(lower));
    REQUIRE(op_v.contains(upper));
    REQUIRE(op_r.lb() >= op_v.lb());
    REQUIRE(op_r.ub() <= op_r.ub());
}
