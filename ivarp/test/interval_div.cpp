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
// Created by Phillip Keldenich on 16.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

static const auto divide = args::x0 / args::x1;

TEST_CASE_TEMPLATE("[ivarp][number] Simple test cases for interval division", NumberType, float, double, Rational) {
    using IV = Interval<NumberType>;

    IV exact{.5f, .5f};
    exact /= exact; // NOLINT deliberate self-division to check for bugs with that.
    REQUIRE(!exact.possibly_undefined());
    REQUIRE(exact.same(IV{1,1}));

    IV exact2{.25f, .5f};
    exact2 /= exact2; // NOLINT
    REQUIRE(!exact.possibly_undefined());
    REQUIRE(exact2.same(IV{.5f, 2.f}));
}

TEST_CASE_TEMPLATE("[ivarp][number] Division by zero test cases", NumberType, float, double, Rational) {
    using IV = Interval<NumberType>;

    // division by exact 0
    REQUIRE((IV{-1,1} / IV{0,0}).same(IV{-infinity,infinity, true}));

    // division by contained zero
    REQUIRE((IV{0,1} / IV{-infinity,infinity}).same(IV{-infinity,infinity,true}));
    REQUIRE((IV{0,1} / IV{-1,1}).same(IV{-infinity,infinity,true}));
    REQUIRE((IV{-1,-.5} / IV{-1,1}).same(IV{-infinity,infinity,true}));
    REQUIRE((IV{-1,-.5} / IV{ 0,2}).same(IV{-infinity, -.25, true}));
    REQUIRE((IV{-1,-.5} / IV{-2,0}).same(IV{.25,infinity,true}));

    // division by one-sided contained zero
    REQUIRE((IV{-1,1} / IV{0,2}).same(IV{-infinity,infinity,true}));
    REQUIRE((IV{1,2} / IV{0,2}).same(IV{0.5f, infinity, true}));
    REQUIRE((IV{1,2} / IV{-2,0}).same(IV{-infinity, -.5f, true}));
}

TEST_CASE_TEMPLATE("[ivarp][number] RandomIntervalDivisionTestCase", NT, float, double, Rational) {
    const IDouble ranges[] = {
            {-1000., -900.},
            {900., 1000.},
            {-0.9, -0.2},
            {0.2, 0.9}
    };
    const IDouble ranges_mixed[] = {
            {-1000., 1000.},
            {-100., 100.},
            {-0.2, 0.2}
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

    random_test_binary_operator<NT>(divide, std::begin(ranges), std::end(ranges),
                                    std::begin(width_classes), std::end(width_classes));
    random_test_binary_operator<NT>(divide, std::begin(ranges_mixed), std::end(ranges_mixed),
                                    std::begin(ranges), std::end(ranges),
                                    std::begin(width_classes), std::end(width_classes));
}

TEST_CASE_TEMPLATE("[ivarp][number] Interval division with infinite bounds", NT, float, double, Rational) {
    using IV = Interval<NT>;

    IV ii{-infinity,infinity}, iit{-infinity,infinity,true};
    IV ip{-infinity, 4};
    IV mi{-4, infinity};
    IV pi{4, infinity};
    IV im{-infinity,-4};
    IV mp{ -4, 8 };

    REQUIRE_SAME(ii / ii, iit);
    REQUIRE_SAME(ii / ip, iit);
    REQUIRE_SAME(ii / mi, iit);
    REQUIRE_SAME(ii / mp, iit);
    REQUIRE_SAME(ii / pi, ii);
    REQUIRE_SAME(ii / im, ii);

    REQUIRE_SAME(im / pi, IV(-infinity,0));
    REQUIRE_SAME(im / im, IV(0,infinity));
    REQUIRE_SAME(pi / pi, IV(0,infinity));
    REQUIRE_SAME(pi / im, IV(-infinity,0));

    REQUIRE_SAME(mp / pi, IV(-1, 2));
    REQUIRE_SAME(mp / im, IV(-2, 1));
}
