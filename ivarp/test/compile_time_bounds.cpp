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
// Created by Phillip Keldenich on 10.01.20.
//

#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;
using namespace ivarp::fixed_point_bounds;

namespace {
    namespace statictest {
        auto bounded_pred = known_bounds<PredicateBounds<false,true>>(args::x0 < args::x1);
        auto bounded_expr = known_bounds<ExpressionBounds<min_bound(), max_bound()>>(args::x0 * args::x1);
        using BPT = decltype(bounded_pred);
        using BET = decltype(bounded_expr);
        static_assert(impl::IsBounded<BPT>::value, "Should be bounded!");
        static_assert(impl::IsBounded<BET>::value, "Should be bounded!");
        using PT = StripBounds<BPT>;
        using ET = StripBounds<BET>;
        static_assert(!impl::IsBounded<PT>::value, "Should not be bounded!");
        static_assert(!impl::IsBounded<ET>::value, "Should not be bounded!");
    }
}

static void test_precision(std::int64_t x1, std::int64_t x2) {
    std::int64_t al = fp_add_rd(x1, x2);
    std::int64_t au = fp_add_ru(x1, x2);
    std::int64_t ml = fp_mul_rd(x1, x2);
    std::int64_t mu = fp_mul_ru(x1, x2);
    REQUIRE(al <= au);
    REQUIRE(au - al <= 1);
    REQUIRE(ml <= mu);
    REQUIRE(mu - ml <= 1);

    if(x2 != 0) {
        std::int64_t dl = fp_div_rd(x1, x2);
        std::int64_t du = fp_div_ru(x1, x2);
        REQUIRE(dl <= du);
        REQUIRE(du - dl <= 1);
    }
}

TEST_CASE("[ivarp][compile time bounds] add/sub/mul/div random") {
    for(int k = 0; k < 100000; ++k) {
        std::int64_t x11 = random_int(-1000 * denom(), 1000 * denom());
        std::int64_t x12 = random_int(-1000 * denom(), 1000 * denom());
        std::int64_t x21 = random_int(-1000 * denom(), 1000 * denom());
        std::int64_t x22 = random_int(-1000 * denom(), 1000 * denom());
        test_precision(x11, x12);
        test_precision(x11, x21);
        test_precision(x11, x22);
        test_precision(x12, x21);
        test_precision(x12, x22);
        test_precision(x21, x22);

        std::int64_t l1 = min(x11, x12);
        std::int64_t l2 = min(x21, x22);
        std::int64_t u1 = max(x11, x12);
        std::int64_t u2 = max(x21, x22);

        Rational rx1 = (fp_to_rational(l1) + fp_to_rational(u1)) / 2;
        Rational rx2 = (fp_to_rational(l2) + fp_to_rational(u2)) / 2;

        IRational ir1{fp_to_rational(l1), fp_to_rational(u1)};
        IRational ir2{fp_to_rational(l2), fp_to_rational(u2)};
        IRational a12 = ir1 + ir2;
        IRational s12 = ir1 - ir2;
        IRational m12 = ir1 * ir2;
        IRational d12 = ir1 / ir2;

        std::int64_t a12l = fp_add_rd(l1, l2);
        std::int64_t a12u = fp_add_ru(u1, u2);
        IRational ra12{fp_to_rational(a12l), fp_to_rational(a12u)};
        std::int64_t s12l = fp_add_rd(l1, -u2);
        std::int64_t s12u = fp_add_ru(u1, -l2);
        IRational rs12{fp_to_rational(s12l), fp_to_rational(s12u)};
        std::int64_t m12l = fp_iv_mul_lb(l1, u1, l2, u2);
        std::int64_t m12u = fp_iv_mul_ub(l1, u1, l2, u2);
        IRational rm12{fp_to_rational(m12l), fp_to_rational(m12u)};
        std::int64_t d12l = fp_iv_div_lb(l1, u1, l2, u2);
        std::int64_t d12u = fp_iv_div_ub(l1, u1, l2, u2);
        IRational rd12{fp_to_rational(d12l), fp_to_rational(d12u)};
        if(!is_lb(d12l)) {
            rd12.set_lb(-infinity);
        }
        if(!is_ub(d12u)) {
            rd12.set_ub(infinity);
        }

        /// Consistency of lb vs ub.
        REQUIRE(a12l <= a12u);
        REQUIRE(s12l <= s12u);
        REQUIRE(m12l <= m12u);
        REQUIRE(d12l <= d12u);

        /// Division consistency.
        if(d12l > min_bound() && d12u < max_bound()) {
            REQUIRE((u2 < 0 || l2 > 0));
        } else {
            REQUIRE(l2 <= 0);
            REQUIRE(u2 >= 0);
        }

        /// Consistency vs. mid-point.
        REQUIRE(ra12.contains(rx1 + rx2));
        REQUIRE(rs12.contains(rx1 - rx2));
        REQUIRE(rm12.contains(rx1 * rx2));
        if(rx2 != 0) {
            REQUIRE(rd12.contains(rx1 / rx2));
        }

        /// Consistency vs. rational bounds.
        REQUIRE(ra12.contains(a12.lb()));
        REQUIRE(ra12.contains(a12.ub()));
        REQUIRE(rs12.contains(s12.lb()));
        REQUIRE(rs12.contains(s12.ub()));
        REQUIRE(rm12.contains(m12.lb()));
        REQUIRE(rm12.contains(m12.ub()));
        if(d12.finite_lb()) {
            REQUIRE(rd12.contains(d12.lb()));
        } else {
            REQUIRE(!rd12.finite_lb());
        }
        if(d12.finite_ub()) {
            REQUIRE(rd12.contains(d12.ub()));
        } else {
            REQUIRE(!rd12.finite_ub());
        }
    }
}

TEST_CASE("[ivarp][compile time bounds] sqrt random") {
    for(int k = 0; k < 1000; ++k) {
        std::int64_t i = random_int(0, denom());
        Rational ri = fp_to_rational(i);
        std::int64_t rd = fp_sqrt_rd(i);
        std::int64_t ru = fp_sqrt_ru(i);
        Rational rrd = fp_to_rational(rd);
        Rational rru = fp_to_rational(ru);
        REQUIRE(rd <= ru);
        REQUIRE(fp_mul_ru(rd,rd) <= i);
        REQUIRE(fp_mul_rd(ru,ru) >= i);
        REQUIRE(rrd*rrd <= ri);
        REQUIRE(rru*rru >= ri);
    }
    for(int k = 0; k < 10000; ++k) {
        std::int64_t i = random_int(denom(), max_bound()-1);
        Rational ri = fp_to_rational(i);
        std::int64_t rd = fp_sqrt_rd(i);
        std::int64_t ru = fp_sqrt_ru(i);
        Rational rrd = fp_to_rational(rd);
        Rational rru = fp_to_rational(ru);
        REQUIRE(rd <= ru);
        REQUIRE(fp_mul_ru(rd,rd) <= i);
        REQUIRE(fp_mul_rd(ru,ru) >= i);
        REQUIRE(rrd*rrd <= ri);
        REQUIRE(rru*rru >= ri);
    }
}

TEST_CASE("[ivarp][compile time bounds] sqrt fixed") {
    auto b1 = fp_iv_sqrt(int_to_fp(4), int_to_fp(4));
    REQUIRE(b1.lb == int_to_fp(2));
    REQUIRE(b1.ub == int_to_fp(2));
    auto b2 = fp_iv_sqrt(int_to_fp(4)-1, int_to_fp(4)+1);
    REQUIRE(b2.lb < int_to_fp(2));
    REQUIRE(b2.ub > int_to_fp(2));
    REQUIRE(fp_sqrt_ru(max_bound()) == max_bound());
    std::int64_t rd = fp_sqrt_rd(max_bound());
    Rational rrd = fp_to_rational(rd);
    REQUIRE(rrd * rrd < fp_to_rational(max_bound()));
}

TEST_CASE("[ivarp][compile time bounds] comparisons") {
    using B1 = Tuple<ExpressionBounds<-2, -1>, ExpressionBounds<0, 1>>;
    using B2 = Tuple<ExpressionBounds<0, 1>, ExpressionBounds<-2, -1>>;

    using LT = BareType<decltype(args::x0 < args::x1)>;
    using GT = BareType<decltype(args::x0 > args::x1)>;
    using LE = BareType<decltype(args::x0 <= args::x1)>;
    using GE = BareType<decltype(args::x0 >= args::x1)>;

    using B1LT = CompileTimeBounds<LT, B1>;
    static_assert(B1LT::lb, "Wrong bound!");
    static_assert(B1LT::ub, "Wrong bound!");

    using B1GT = CompileTimeBounds<GT, B1>;
    static_assert(!B1GT::lb, "Wrong bound!");
    static_assert(!B1GT::ub, "Wrong bound!");

    using B1LE = CompileTimeBounds<LE, B1>;
    static_assert(B1LE::lb, "Wrong bound!");
    static_assert(B1LE::ub, "Wrong bound!");

    using B1GE = CompileTimeBounds<GE, B1>;
    static_assert(!B1GE::lb, "Wrong bound!");
    static_assert(!B1GE::ub, "Wrong bound!");

    using B2LT = CompileTimeBounds<LT, B2>;
    static_assert(!B2LT::lb, "Wrong bound!");
    static_assert(!B2LT::ub, "Wrong bound!");

    using B2GT = CompileTimeBounds<GT, B2>;
    static_assert(B2GT::lb, "Wrong bound!");
    static_assert(B2GT::ub, "Wrong bound!");

    using B2LE = CompileTimeBounds<LE, B2>;
    static_assert(!B2LE::lb, "Wrong bound!");
    static_assert(!B2LE::ub, "Wrong bound!");

    using B2GE = CompileTimeBounds<LE, B2>;
    static_assert(!B2GE::lb, "Wrong bound!");
    static_assert(!B2GE::ub, "Wrong bound!");
}

struct StaticTest1Bounds {
    static constexpr std::int64_t lb = int_to_fp(49);
    static constexpr std::int64_t ub = int_to_fp(49);
};

struct StaticTest2Bounds {
    static constexpr std::int64_t lb = int_to_fp(49)-1;
    static constexpr std::int64_t ub = int_to_fp(49)+1;
};

using StaticTest1 = SqrtEvalBounds<StaticTest1Bounds>;
static_assert(StaticTest1::lb == int_to_fp(7), "Wrong value!");
static_assert(StaticTest1::ub == int_to_fp(7), "Wrong value!");

using StaticTest2 = SqrtEvalBounds<StaticTest2Bounds>;
static_assert(StaticTest2::lb < int_to_fp(7), "Wrong value!");
static_assert(StaticTest2::ub > int_to_fp(7), "Wrong value!");
