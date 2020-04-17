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
// Created by Phillip Keldenich on 28.01.20.
//

#include "ivarp/refactor_constraint_system/bound_and_simplify.hpp"
#include "ivarp/bound_dependency_analysis.hpp"
#include "test_util.hpp"
#include <tuple>

using namespace ivarp;

TEST_CASE("[ivarp][bound dependency analysis] Simple test: addition") {
    using NoBounds = Tuple<fixed_point_bounds::Unbounded, fixed_point_bounds::Unbounded, fixed_point_bounds::Unbounded>;
    auto f1 = bound_and_simplify<NoBounds>(args::x0 + args::x1);
    BoundDependencies f1d0 = compute_bound_dependencies<decltype(f1), 0>();
    REQUIRE(f1d0 == (BoundDependencies{true,false,false,true}));
    BoundDependencies f1d1 = compute_bound_dependencies<decltype(f1), 1>();
    REQUIRE(f1d1 == (BoundDependencies{true,false,false,true}));
    BoundDependencies f1d2 = compute_bound_dependencies<decltype(f1), 2>();
    REQUIRE(f1d2 == (BoundDependencies{false,false,false,false}));
}

TEST_CASE("[ivarp][bound dependency analysis] Simple test: subtraction") {
    using NoBounds = Tuple<fixed_point_bounds::Unbounded, fixed_point_bounds::Unbounded, fixed_point_bounds::Unbounded>;
    auto f1 = bound_and_simplify<NoBounds>(args::x0 + (-args::x1) - args::x2);
    BoundDependencies f1d0 = compute_bound_dependencies<decltype(f1), 0>();
    REQUIRE(f1d0 == (BoundDependencies{true,false,false,true}));
    BoundDependencies f1d1 = compute_bound_dependencies<decltype(f1), 1>();
    REQUIRE(f1d1 == (BoundDependencies{false,true,true,false}));
    BoundDependencies f1d2 = compute_bound_dependencies<decltype(f1), 2>();
    REQUIRE(f1d2 == (BoundDependencies{false,true,true,false}));
    REQUIRE(f1(22.0, 15.0, 16.5) == -9.5);
}

namespace {
    template<typename B1, typename B2>
        std::pair<BoundDependencies,BoundDependencies> test_mul()
    {
        auto fmul = bound_and_simplify<Tuple<B1,B2>>(args::x0 * args::x1);
        for(int i = 0; i < 16; ++i) {
            std::int64_t a = random_int(B1::lb, B1::ub);
            std::int64_t b = random_int(B2::lb, B2::ub);
            Rational ra{fixed_point_bounds::fp_to_rational(a)}, rb{fixed_point_bounds::fp_to_rational(b)};
            IDouble iva = convert_number<IDouble>(ra), ivb = convert_number<IDouble>(rb);
            REQUIRE(fmul(ra, rb) == ra * rb);
            REQUIRE(fmul(iva,ivb).contains(ra*rb));
        }

        return { 
            compute_bound_dependencies<decltype(fmul),0>(),
            compute_bound_dependencies<decltype(fmul),1>()
        };
    }

    template<typename B1, typename B2>
        std::pair<BoundDependencies,BoundDependencies> test_div()
    {
        auto fdiv = bound_and_simplify<Tuple<B1,B2>>(args::x0 / args::x1);
        static_assert((B1::lb < 0 || B2::lb <= 0 || decltype(fdiv)::lb >= 0), "Bound error!");

        for(int i = 0; i < 16; ++i) {
            std::int64_t a = random_int(B1::lb, B1::ub);
            std::int64_t b = random_int(B2::lb, B2::ub);
            while(b == 0) {
                b = random_int(B2::lb, B2::ub);
            }
            Rational ra{fixed_point_bounds::fp_to_rational(a)}, rb{fixed_point_bounds::fp_to_rational(b)};
            IDouble iva = convert_number<IDouble>(ra), ivb = convert_number<IDouble>(rb);
            REQUIRE(fdiv(ra, rb) == ra / rb);
            REQUIRE(fdiv(iva,ivb).contains(ra/rb));
        }

        return {
            compute_bound_dependencies<decltype(fdiv),0>(),
            compute_bound_dependencies<decltype(fdiv),1>()
        };
    }

    using BP = ExpressionBounds<1, fixed_point_bounds::int_to_fp(5)>;
    using BN = ExpressionBounds<-fixed_point_bounds::int_to_fp(6), -fixed_point_bounds::int_to_fp(4)>;
    using BM = ExpressionBounds<-fixed_point_bounds::int_to_fp(42), fixed_point_bounds::int_to_fp(23)>;

    BoundDependencies bpos{true,false,false,true};
    BoundDependencies bneg{false,true,true,false};
    BoundDependencies blb{true,false,true,false};
    BoundDependencies bub{false,true,false,true};
    BoundDependencies ball{true,true,true,true};
}

TEST_CASE("[ivarp][bound dependency analysis] Simple test: multiplication") {
    // Check every possible combination.
    BoundDependencies b0, b1;
    std::tie(b0,b1) = test_mul<BP,BP>();
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == bpos);
    std::tie(b0,b1) = test_mul<BP,BN>();
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == bpos);
    std::tie(b0,b1) = test_mul<BP,BM>(); // lb = ub * lb, ub = ub * ub
    REQUIRE(b0 == bub);
    REQUIRE(b1 == bpos);
    std::tie(b0,b1) = test_mul<BN,BP>();
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == bneg);
    std::tie(b0,b1) = test_mul<BN,BN>();
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == bneg);
    std::tie(b0,b1) = test_mul<BN,BM>(); // lb = lb * ub, ub = lb * lb
    REQUIRE(b0 == blb);
    REQUIRE(b1 == bneg);
    std::tie(b0,b1) = test_mul<BM,BP>(); // lb = lb * ub, ub = ub * ub
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == bub);
    std::tie(b0,b1) = test_mul<BM,BN>(); // lb = ub * lb, ub = lb * lb
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == blb);
    std::tie(b0,b1) = test_mul<BM,BM>();
    REQUIRE(b0 == ball);
    REQUIRE(b1 == ball);
}

TEST_CASE("[ivarp][bound dependency analysis] Simple test: division") {
    BoundDependencies b0, b1;
    std::tie(b0,b1) = test_div<BP,BP>(); // lb = lb/ub, ub = ub/lb
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == bneg);
    std::tie(b0,b1) = test_div<BP,BN>(); // lb = ub/ub, ub = lb/lb
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == bneg);
    std::tie(b0,b1) = test_div<BP,BM>();
    REQUIRE(b0 == ball);
    REQUIRE(b1 == ball);
    std::tie(b0,b1) = test_div<BN,BP>(); // lb = lb/lb, ub = ub/ub
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == bpos);
    std::tie(b0,b1) = test_div<BN,BN>(); // lb = ub/lb, ub = lb/ub
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == bpos);
    std::tie(b0,b1) = test_div<BN,BM>();
    REQUIRE(b0 == ball);
    REQUIRE(b1 == ball);
    std::tie(b0,b1) = test_div<BM,BP>(); // lb = lb/lb, ub = ub/lb
    REQUIRE(b0 == bpos);
    REQUIRE(b1 == blb);
    std::tie(b0,b1) = test_div<BM,BN>(); // lb = ub/ub, ub = lb/ub
    REQUIRE(b0 == bneg);
    REQUIRE(b1 == bub);
    std::tie(b0,b1) = test_div<BM,BM>();
    REQUIRE(b0 == ball);
    REQUIRE(b1 == ball);
}
