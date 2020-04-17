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
// Created by Phillip Keldenich on 22.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

auto root = sqrt(args::x0);
auto sqr = args::x0 * args::x0;

TEST_CASE_TEMPLATE("[ivarp][math_fn] Interval square root with invalid inputs", NumberType, float, double, Rational) {
    using IV = Interval<NumberType>;
    IV all{-infinity,infinity};
    IV neginf{-infinity,4, false};
    IV neg{-4,4};
    IV negneg{-4, -1};
    REQUIRE_SAME(root(all), IV(0, infinity, true));
    REQUIRE_SAME(root(neginf), IV(0, 2, true));
    REQUIRE_SAME(root(neg), IV(0,2,true));
    REQUIRE_SAME(root(negneg), IV(-infinity,infinity,true));
}

static inline bool isnext(const Rational&, const Rational&) { return true; }
template<typename F>
static inline std::enable_if_t<std::is_floating_point<F>::value, bool> isnext(F lb, F ub) {
    return std::nextafter(lb, std::numeric_limits<F>::infinity()) == ub;
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Interval square root test against square", NumberType, float, double, Rational) {
    // must not contain rational squares
    NumberType values[] = {
        0.5f, 0.75f, 2, 5, 7, 10.25f, 1000
    };
    for(const NumberType& n : values) {
        Interval<NumberType> r = root(Interval<NumberType>(n));
        REQUIRE(r.lb() < r.ub());
        REQUIRE(isnext(r.lb(), r.ub()));
        REQUIRE(!r.possibly_undefined());
        Interval<NumberType> rsq = sqr(r);
        REQUIRE(rsq.lb() < n);
        REQUIRE(rsq.ub() > n);
    }
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Interval square root random test", NumberType, float, double, Rational) {
    using IV = Interval<NumberType>;
    constexpr unsigned NUM_SQRT_RANDOM_TESTS = 65536;

    IDouble width_classes[] = {
        {0.,  0.001},
        {0.,  0.1},
        {0.1, 0.5},
        {0.4, 0.8},
        {1.,  2.},
        {2.,  50.}
    };
    IDouble range{0,100};

    for(unsigned j = 0; j < NUM_SQRT_RANDOM_TESTS; ++j) {
        IDouble rid = random_interval(range, std::begin(width_classes), std::end(width_classes));
        IV ri = convert_number<IV>(rid);
        IV sqrtri = root(ri);
        REQUIRE(!sqrtri.possibly_undefined());
        REQUIRE(sqrtri.lb() >= 0);
        IV sqrtrisq = sqr(sqrtri);
        REQUIRE(sqrtrisq.lb() <= ri.lb());
        REQUIRE(sqrtrisq.ub() >= ri.ub());
    }
}
