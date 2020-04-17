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

#define _USE_MATH_DEFINES
#include <doctest/doctest_fixed.hpp>
#include <cmath>
#include <climits>
#include <cfloat>
#include "ivarp/math_fn.hpp"
#include "ivarp/rounding.hpp"
#include "test_util.hpp"

using namespace ivarp;

static const auto sine = sin(args::x0);
static const auto cosine = cos(args::x0);
static_assert(std::is_same<std::decay_t<decltype(sine(std::declval<Rational&>()))>, IRational>::value, "Promotion error?");
static_assert(std::is_same<std::decay_t<decltype(cosine(std::declval<Rational&>()))>, IRational>::value, "Promotion error?");

TEST_CASE_TEMPLATE("[ivarp][math_fn] Sine (float/double/rational intervals)", FloatType, float, double, Rational) {
    using IV = Interval<FloatType>;
    REQUIRE(sine(IV{0.f}).same(IV{0.f}));
    REQUIRE(sine(IV{0.f,6.3f}).same(IV{-1.f,1.f}));
    REQUIRE(sine(IV{0.f,0.1f}).lb() == 0.f);
    REQUIRE(cosine(IV{0.f}).same(IV{1.f}));
    REQUIRE(cosine(IV{0.f,6.3f}).same(IV{-1.f,1.f}));

    IV close_to_zero{std::nextafterf(0.f,-1.f), std::nextafterf(0.f,1.f)};
    REQUIRE(sine(close_to_zero).lb() < 0.f);
    REQUIRE(sine(close_to_zero).ub() > 0.f);
    REQUIRE(cosine(close_to_zero).ub() == 1.f);
    REQUIRE(cosine(close_to_zero).lb() < 1.f);

    IV close_to_pi_2{std::nextafterf((float)M_PI_2, -1.f), std::nextafterf((float)M_PI_2, 2.f)};
    IV s = sine(close_to_pi_2);
    IV c = cosine(close_to_pi_2);
    REQUIRE(s.ub() == 1);
    REQUIRE(s.lb() < 1);
    REQUIRE(s.lb() > 0.99f);
    REQUIRE(c.ub() > 0);
    REQUIRE(c.ub() < 0.01f);
    REQUIRE(c.lb() < 0);
    REQUIRE(c.lb() > -0.01f);

    float pi_6_approx = 0.523598775598298873077107230546583814032861566562517636829f;
    IV close_to_pi_6{ std::nextafterf(pi_6_approx, 0.f), std::nextafterf(pi_6_approx, 1.f) };
    s = sine(close_to_pi_6);
    c = cosine(close_to_pi_6);
    REQUIRE(s.ub() > 0.5f);
    REQUIRE(s.lb() < 0.5f);
    REQUIRE(s.lb() > 0.49f);
    REQUIRE(s.ub() < 0.51f);
    REQUIRE(c.ub() > 0.866f);
    REQUIRE(c.ub() < 0.867f);
    REQUIRE(c.lb() < c.ub());

    IV cross_pi_2{M_PI_4, M_PI_2+M_PI_4};
    s = sine(cross_pi_2);
    c = cosine(cross_pi_2);
    REQUIRE(s.lb() < 0.7072f);
    REQUIRE(s.lb() > 0.707f);
    REQUIRE(s.ub() == 1);
    REQUIRE(c.lb() < -0.707f);
    REQUIRE(c.lb() > -0.7072f);
    REQUIRE(c.ub() < 0.7072f);
    REQUIRE(c.ub() > 0.707f);

    IV all{-infinity,infinity};
    REQUIRE(!is_finite(all));
    REQUIRE(sine(all).same(IV{-1,1}));
    REQUIRE(cosine(all).same(IV{-1,1}));
    REQUIRE(is_finite(sine(all)));
    REQUIRE(is_finite(cosine(all)));

    IV half{0.5f};
    s = sine(half);
    c = cosine(half);
    REQUIRE(s.lb() < s.ub());
    REQUIRE(s.lb() > 0.479f);
    REQUIRE(s.ub() < 0.48f);
    REQUIRE(c.lb() < 0.8776f);
    REQUIRE(c.ub() > 0.8775f);
    REQUIRE(is_finite(s));
    REQUIRE(is_finite(c));

    IV cross_0{-std::nextafterf((float)M_PI_4, 5.f), std::nextafterf((float)M_PI_4, 5.f) };
    s = sine(cross_0);
    c = cosine(cross_0);
    REQUIRE(s.lb() < -0.7071f);
    REQUIRE(s.lb() > -0.7072f);
    REQUIRE(c.lb() < 0.7072f);
    REQUIRE(c.ub() > 0.7071f);
    REQUIRE(s.ub() > 0.7071f);
    REQUIRE(s.ub() < 0.7072f);
    REQUIRE(is_finite(s));
    REQUIRE(is_finite(c));

    IV cross_min_max{M_PI_4, 2*M_PI-M_PI_4};
    s = sine(cross_min_max);
    REQUIRE(s.same(IV{-1,1}));
    REQUIRE(is_finite(s));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Cosine/Sine (float/double, too wide)", FT, float, double) {
    using IV = Interval<FT>;
    using IR = IRational;

    IV too_wide1{
        std::nextafter(FT(2*M_PI), -std::numeric_limits<FT>::infinity()),
        std::nextafter(FT(4*M_PI), std::numeric_limits<FT>::infinity())
    };
    IR rtoo_wide1 = convert_number<IR>(too_wide1);
    IV too_wide2{
        std::nextafter(FT(2*M_PI)+1.25f, -std::numeric_limits<FT>::infinity()),
        std::nextafter(FT(4*M_PI)+1.25f, std::numeric_limits<FT>::infinity())
    };
    IR rtoo_wide2 = convert_number<IR>(too_wide2);
    REQUIRE_SAME(sine(too_wide1), IV(-1,1));
    REQUIRE_SAME(sine(rtoo_wide1), IR(-1,1));
    REQUIRE_SAME(cosine(too_wide1), IV(-1,1));
    REQUIRE_SAME(cosine(rtoo_wide1), IR(-1,1));
    REQUIRE_SAME(sine(too_wide2), IV(-1,1));
    REQUIRE_SAME(sine(rtoo_wide2), IR(-1,1));
    REQUIRE_SAME(cosine(too_wide2), IV(-1,1));
    REQUIRE_SAME(cosine(rtoo_wide2), IR(-1,1));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Cosine/Sine (random intervals, float/double vs. rational)", FloatType, float, double) {
    constexpr unsigned NUM_SINE_RANDOM_TESTS = 65536;
    Interval<FloatType> width_classes[] = {
            { 0.f, 0.001f },
            { 0.f, 0.1f  },
            { 0.1f, 0.5f },
            { 0.4f, 0.8f },
            { 1.f, 2.f }
    };
    Interval<FloatType> range{-6.5f, 6.5f};
    for(unsigned j = 0; j < NUM_SINE_RANDOM_TESTS; ++j) {
        Interval<FloatType> rnd = random_interval(range, std::begin(width_classes), std::end(width_classes));
        IRational rndr = convert_number<IRational>(rnd);
        Rational c = random_point(rndr);
        Interval<FloatType> s = sine(rnd);
        Interval<FloatType> o = cosine(rnd);
        Interval<Rational> rs = sine(c);
        Interval<Rational> ro = cosine(c);
        REQUIRE(lb(rs) >= lb(s));
        REQUIRE(ub(rs) <= ub(s));
        REQUIRE(lb(ro) >= lb(o));
        REQUIRE(ub(ro) <= ub(o));
    }
}

struct CustomContext {
    using NumberType = IRational;
    static constexpr bool analyze_monotonicity = false;
    static constexpr unsigned monotonicity_derivative_levels = 0;
    static constexpr unsigned irrational_precision = 4096; // ~ about 1230 decimal places
};

TEST_CASE("[ivarp][math_fn] Sine with high-precision context") {
    std::tuple<Rational, Rational, const char*, const char*, const char*, const char*> cases[] = {
        {
            5, 5,
            // expected LB range
            "-958924274663138468893154406155993973352461543964601778131672454235102558086559603076995955429532866596530638462/"
            "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "-958924274663138468893154406155993973352461543964601778131672454235102558086559603076995955429532866596530638460/"
            "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            // expected UB range
            "-958924274663138468893154406155993973352461543964601778131672454235102558086559603076995955429532866596530638462/"
            "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "-958924274663138468893154406155993973352461543964601778131672454235102558086559603076995955429532866596530638460/"
            "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        },
        { 4.5, 8.5, "-1", "-1", "1", "1" },
        { 4.75, 9,
          // expected LB range
          "-999292788975377944734270755597079263247762344906818509104960880011953734503240377853903983317698482694049152526/"
          "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          // expected UB range
          "-999292788975377944734270755597079263247762344906818509104960880011953734503240377853903983317698482694049152524/"
          "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
          "1", "1"
        },
        { 4.75, 11.5, "-1", "-1", "1", "1"}
    };

    for(const auto& c : cases) {
        IRational result = sine.evaluate<CustomContext>(IRational{std::get<0>(c), std::get<1>(c)});
        Rational expect_lb_ge{std::get<2>(c)};
        expect_lb_ge.canonicalize();
        Rational expect_lb_le{std::get<3>(c)};
        expect_lb_le.canonicalize();
        Rational expect_ub_ge{std::get<4>(c)};
        expect_ub_ge.canonicalize();
        Rational expect_ub_le{std::get<5>(c)};
        expect_ub_le.canonicalize();
        REQUIRE(result.lb() >= expect_lb_ge);
        REQUIRE(result.lb() <= expect_lb_le);
        REQUIRE(result.ub() >= expect_ub_ge);
        REQUIRE(result.ub() <= expect_ub_le);
        REQUIRE(!result.possibly_undefined());
    }
}
