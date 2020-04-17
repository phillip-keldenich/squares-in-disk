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
// Created by Phillip Keldenich on 14.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

static const auto multiply = args::x0 * args::x1;

TEST_CASE_TEMPLATE("[ivarp][math_fn] Test multiplication function correctness", NT, IFloat, IDouble, IRational) {
    REQUIRE(NT{25}.same(multiply(NT{5}, NT{5})));
    REQUIRE(NT{2}.same(multiply(NT{-1}, NT{-2})));
    REQUIRE(NT{-4}.same(multiply(NT{-2}, NT{2})));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Simple interval multiplication tests (float/double)", FloatType, float, double) {
    const FloatType ma = std::numeric_limits<FloatType>::max();
    const FloatType inf = std::numeric_limits<FloatType>::infinity();
    using IV = Interval<FloatType>;

    IV exact{0.5f,0.5f};

    ivarp::sqrt<DefaultContextWithNumberType<Interval<FloatType>>>(exact);
    exact *= 2;
    IV exact2 = exact * 2;

    REQUIRE(exact.same(IV{1,1}));
    REQUIRE(exact2.same(IV{2,2}));

    IV large{ma, inf};
    REQUIRE((large*2).same(IV{ma, inf}));
    REQUIRE((large*0).same(IV{0, 0}));
    REQUIRE((large*IV{0,std::nextafter(FloatType(0), FloatType(1))}).same(IV{0, inf}));
    REQUIRE((large*IV{-1,1}).same(IV{-inf,inf}));
    IV large_mixed{-ma,ma};
    IV small_mixed{std::nextafter(FloatType(0), FloatType(-1)),
                   std::nextafter(FloatType(0), FloatType(1))};
    REQUIRE((large_mixed*2).same(IV{-inf,inf}));
    IV res_mixed = large_mixed * small_mixed;
    REQUIRE(lb(res_mixed) < 0);
    REQUIRE(ub(res_mixed) > 0);
    REQUIRE(lb(res_mixed) == -ub(res_mixed));
    IV asym_mixed{-2, 1};
    REQUIRE((asym_mixed * ma).same(IV{-inf, ma}));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Random interval multiplications", NT, float, double, Rational) {
    const IDouble ranges[] = {
            {-100., 100.},
            {-1000., -900. },
            { 900., 1000. }
    };
    const IDouble width_classes[] = {
            {0, 0.1},
            {0, 0.5},
            {0.1,0.3},
            {0.5,2.5},
            {0.5,5.},
            {5.,20.},
            {20., 50.}
    };
    random_test_binary_operator<NT>(multiply, std::begin(ranges), std::end(ranges),
                                    std::begin(width_classes), std::end(width_classes));
}
