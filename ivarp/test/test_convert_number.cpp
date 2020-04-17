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
// Created by Phillip Keldenich on 2019-10-04.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"
#include <random>
#include <cstdint>
#include <climits>
#include <iostream>
#include <iomanip>

#include "test_util.hpp"

namespace ivarp {
namespace impl {
    // STATIC ASSERTION TESTS FOR MaxRank
    static_assert(std::is_same<MaxRank<double, float>::type, double>::value, "MaxRank(double,float) != double!");
    static_assert(std::is_same<MaxRank<Rational, float>::type, Rational>::value, "MaxRank(Rational,float) != Rational");
    static_assert(std::is_same<MaxRank<Rational, double>::type, Rational>::value, "MaxRank(Rational,double) != Rational");
    static_assert(std::is_same<MaxRank<Rational, Rational>::type, Rational>::value, "MaxRank(Rational,Rational) != Rational");
    static_assert(std::is_same<MaxRank<Rational, IFloat>::type, IFloat>::value, "MaxRank(Rational,IFloat) != IFloat");
    static_assert(std::is_same<MaxRank<Rational, IDouble>::type, IDouble>::value, "MaxRank(Rational,IDouble) != IDouble");
    static_assert(std::is_same<MaxRank<IFloat, IDouble>::type, IDouble>::value, "MaxRank(IFloat,IDouble) != IDouble");
    static_assert(std::is_same<MaxRank<IFloat, IFloat>::type, IFloat>::value, "MaxRank(IFloat,IFloat) != IFloat");
    static_assert(std::is_same<MaxRank<IRational, IDouble>::type, IRational>::value, "MaxRank(IRational,IDouble) != IRational");

    // STATIC ASSERTION TESTS FOR NumberTypePromotion
    static_assert(std::is_same<Promote<float, double, double, float, float>, double>::value, "Should be double!");
    static_assert(std::is_same<Promote<float, double, Rational, IFloat>, IFloat>::value, "Should be IFloat!");
    static_assert(std::is_same<Promote<float, IFloat, IRational>, IRational>::value, "Should be IRational!");
}
}

template<typename FloatType, typename RNG> long random_representable_int(RNG& rng) {
    constexpr unsigned digits = std::min(unsigned(sizeof(long)*CHAR_BIT-1), unsigned(std::numeric_limits<FloatType>::digits));
    constexpr long max = long((1ul << digits) - 1ul);
    std::uniform_int_distribution<long> dist(-max, max);
    return dist(rng);
}

template<typename FloatType, typename RNG> long random_odd_representable_int(RNG& rng) {
    return random_representable_int<FloatType>(rng) | 1; // NOLINT
}

template<typename RNG> long random_exact_divisor(RNG& rng) {
    constexpr auto max = unsigned(sizeof(long) * CHAR_BIT - 1);
    return 1l << std::uniform_int_distribution<unsigned>(0, max)(rng); // NOLINT
}

TEST_CASE("[ivarp][number] Exact conversion Rational -> IFloat") {
    constexpr unsigned NUM_RANDOM_CONVERSION_TESTS = 65536;
    for(unsigned j = 0; j < NUM_RANDOM_CONVERSION_TESTS; ++j) {
        long i = random_representable_int<float>(ivarp::rng);
        long d = random_exact_divisor(ivarp::rng);
        float fi = static_cast<float>(i); // NOLINT
        float di = static_cast<float>(d); // NOLINT
        REQUIRE(static_cast<long>(fi) == i);
        REQUIRE(static_cast<long>(di) == d);
        ivarp::Rational r = ivarp::rational(i, d);
        ivarp::IFloat result = ivarp::convert_number<ivarp::IFloat>(r);
        REQUIRE(result.lb() == result.ub());
    }
}

TEST_CASE("[ivarp][number] Exact conversion Rational -> IDouble") {
    constexpr unsigned NUM_RANDOM_CONVERSION_TESTS = 65536;
    for(unsigned j = 0; j < NUM_RANDOM_CONVERSION_TESTS; ++j) {
        long i = random_representable_int<double>(ivarp::rng);
        long d = random_exact_divisor(ivarp::rng);
        double fi = static_cast<double>(i); // NOLINT
        double di = static_cast<double>(d); // NOLINT
        REQUIRE(static_cast<long>(fi) == i);
        REQUIRE(static_cast<long>(di) == d);
        ivarp::Rational r = ivarp::rational(i, d);
        ivarp::IDouble result = ivarp::convert_number<ivarp::IDouble>(r);
        REQUIRE(result.lb() == result.ub());
    }
}

TEST_CASE("[ivarp][number] Inexact conversion Rational -> IFloat/IDouble") {
    int small[] = { 3, 5, 7, 9, 11, 13, 15, 17, 19 };
    for(int i : small) {
        ivarp::Rational r = ivarp::rational(1, i);
        ivarp::IFloat f = ivarp::convert_number<ivarp::IFloat>(r);
        ivarp::IDouble d = ivarp::convert_number<ivarp::IDouble>(r);
        REQUIRE(lb(f) < ub(f));
        REQUIRE(lb(d) < ub(d));
        REQUIRE(std::nextafterf(lb(f), 1.0f) == ub(f));
        REQUIRE(std::nextafter(lb(d), 1.0) == ub(d));
        REQUIRE(lb(f) < r);
        REQUIRE(lb(d) < r);
        REQUIRE(r < ub(f));
        REQUIRE(r < ub(d));
    }

    constexpr unsigned NUM_RANDOM_CONVERSION_TESTS = 65536;
    for(unsigned j = 0; j < NUM_RANDOM_CONVERSION_TESTS; ++j) {
        long fi = random_odd_representable_int<float>(ivarp::rng);
        long long di = random_odd_representable_int<double>(ivarp::rng);
        if(fi == 1 || di == 1) {
            --j; continue;
        }
        ivarp::Rational rf = ivarp::rational(1, fi);
        ivarp::IFloat f = ivarp::convert_number<ivarp::IFloat>(rf);
        ivarp::Rational rd = ivarp::rational(1, di);
        ivarp::IDouble d = ivarp::convert_number<ivarp::IDouble>(rd);
        REQUIRE(lb(f) < ub(f));
        REQUIRE(lb(d) < ub(d));
        REQUIRE(std::nextafterf(lb(f), 1.0f) == ub(f));
        REQUIRE(std::nextafter(lb(d), 1.0) == ub(d));
        REQUIRE(lb(f) < rf);
        REQUIRE(lb(d) < rd);
        REQUIRE(rf < ub(f));
        REQUIRE(rd < ub(d));
    }
}

using namespace ivarp;

template<typename T1, typename T2> static inline Rational distance(const T1& t1, const T2& t2) {
    if(t1 < t2) {
        return Rational(t2) - Rational(t1);
    } else {
        return Rational(t1) - Rational(t2);
    }
}

TEST_CASE_TEMPLATE("[ivarp][number] Inexact conversion (distance bound & intervals)", NT, float, double, Rational) {
    Rational values[] = { Rational{1, 3}, Rational{2,3}, Rational{7, 11},
                          Rational{195, 256}, Rational{17331, 42218}};
    for(const Rational& r : values) {
        NT v = convert_number<NT>(r);
        NT m = convert_number<NT>(-r);
        Rational dv = distance(v, r);
        Rational dn = distance(m, -r);
        REQUIRE(dv < Rational{1,1000});
        REQUIRE(dn < Rational{1,1000});
        Interval<NT> iv = tested_conversion<Interval<NT>>(r);
        Interval<NT> im = tested_conversion<Interval<NT>>(-r);
        REQUIRE(distance(lb(iv), ub(iv)) <= Rational{1,1000});
        REQUIRE(distance(lb(im), ub(im)) <= Rational{1,1000});
    }
}

TEST_CASE_TEMPLATE("[ivarp][number] Extremely small rational conversion", IV, IFloat, IDouble) {
    using FT = typename IV::NumberType;
    const FT actual_min = std::nextafter(FT(0), FT(1));
    Rational tiny{actual_min};
    for(int i = 0; i < 100; ++i) {
        tiny /= 1000000;
    }
    Rational miny = -tiny;
    IV t = convert_number<IV>(tiny);
    IV m = convert_number<IV>(miny);
    REQUIRE(t.lb() == 0);
    REQUIRE((t.ub() == std::numeric_limits<FT>::min() || t.ub() == actual_min));
    REQUIRE(m.ub() == 0);
    REQUIRE((m.lb() == -std::numeric_limits<FT>::min() || m.lb() == -actual_min));
}

TEST_CASE("[ivarp][number] IDouble to IFloat conversion") {
    IDouble large{std::nextafter(std::numeric_limits<double>::max(), 0.) };
    IDouble large2{std::nextafter((double)std::numeric_limits<float>::max(), std::numeric_limits<double>::infinity())};
    IFloat from_large = convert_number<IFloat>(large);
    IFloat from_large2 = convert_number<IFloat>(large2);
    IFloat from_mlarge = convert_number<IFloat>(-large2);

    REQUIRE(from_large.same(from_large2));
    REQUIRE(lb(from_large) == std::numeric_limits<float>::max());
    REQUIRE(ub(from_large) == std::numeric_limits<float>::infinity());
    REQUIRE(lb(from_mlarge) == -std::numeric_limits<float>::infinity());
    REQUIRE(ub(from_mlarge) == -std::numeric_limits<float>::max());
    REQUIRE(from_large.contains(large.lb()));
}

TEST_CASE_TEMPLATE("[ivarp][number] Large integer conversion", NT, IFloat, IDouble) {
    std::uint64_t too_large_2bit_rd = 36028797018963969ull; // in round-to-nearest, this rounds down to 1 << 55 (...68)
    std::uint64_t too_large_2bit_ru = 36028797018963973ull; // in round-to-nearest, this rounds up to ...76
    std::int64_t very_large = 9223372036854775807ll;
    NT vl = convert_number<NT>(very_large);
    NT rd = convert_number<NT>(too_large_2bit_rd);
    NT ru = convert_number<NT>(too_large_2bit_ru);
    std::uint64_t very_large2 = 18446744073709551615ull;
    NT vl2 = convert_number<NT>(very_large2);
    std::uint64_t large_exact = std::uint64_t(1) << 62u;
    NT lex = convert_number<NT>(large_exact);

    REQUIRE_SAME(rd, ru);
    REQUIRE(rd.lb() < rd.ub());
    REQUIRE(exact_less_than(rd.lb(), too_large_2bit_rd));
    REQUIRE(exact_less_than(rd.lb(), too_large_2bit_ru));
    REQUIRE(exact_less_than(too_large_2bit_rd, rd.ub()));
    REQUIRE(exact_less_than(too_large_2bit_ru, rd.ub()));
    REQUIRE(exact_less_than(vl.lb(), very_large));
    REQUIRE(exact_less_than(very_large, vl.ub()));
    REQUIRE(exact_less_than(vl2.lb(), very_large2));
    REQUIRE(exact_less_than(very_large2, vl2.ub()));
    REQUIRE(!exact_less_than(lex.lb(), large_exact));
    REQUIRE(!exact_less_than(large_exact, lex.lb()));
    REQUIRE(lex.lb() == lex.ub());
}

TEST_CASE_TEMPLATE("[ivarp][number] Conversion with undefined flag", NT, IFloat, IDouble) {
    IRational r{11, 15, true};
    auto c = convert_number<NT>(r);
    REQUIRE(c.lb() == 11);
    REQUIRE(c.ub() == 15);
    REQUIRE(c.possibly_undefined());
}
