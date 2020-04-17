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
// Created by Phillip Keldenich on 17.01.20.
//

#include "test_util.hpp"
#include "ivarp/refactor_constraint_system.hpp"

using namespace ivarp;

TEST_CASE("[ivarp][constraint refactoring] Simplify and bound - simple") {
    auto f1 = args::x0 + 2_Z * args::x1;
    using F1Bounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(33), fixed_point_bounds::int_to_fp(34)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(5), fixed_point_bounds::int_to_fp(8)>>;
    auto f1s = bound_and_simplify<F1Bounds>(ivarp::move(f1));
    using F1S = decltype(f1s);

    static_assert(F1S::lb == fixed_point_bounds::int_to_fp(43), "Wrong lb!");
    static_assert(F1S::ub == fixed_point_bounds::int_to_fp(50), "Wrong ub!");
    REQUIRE(f1s(33.0, 5.0) == 43.0);
    REQUIRE(f1s(33.5, 6.5) == 46.5);
    REQUIRE(f1s(34.0, 8.0) == 50.0);
}

TEST_CASE("[ivarp][constraint refactoring] Simplify and bound - comparisons") {
    auto f1 = (args::x0 <= 2_Z * args::x1);
    using F1Bounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(15), fixed_point_bounds::int_to_fp(16)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(8), fixed_point_bounds::int_to_fp(9)>>;

    using F2Bounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(15), fixed_point_bounds::int_to_fp(16)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(8), fixed_point_bounds::int_to_fp(8)>>;

    using F3Bounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(15), fixed_point_bounds::int_to_fp(16)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(7), fixed_point_bounds::int_to_fp(7)>>;

    using F4Bounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(13), fixed_point_bounds::int_to_fp(15)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(7), fixed_point_bounds::int_to_fp(7)>>;

    using F1 = decltype(f1);
    auto f1s = bound_and_simplify<F1Bounds>(F1(f1));
    auto f2s = bound_and_simplify<F2Bounds>(F1(f1));
    auto f3s = bound_and_simplify<F3Bounds>(F1(f1));
    auto f4s = bound_and_simplify<F4Bounds>(F1(f1));

    static_assert(std::is_same<decltype(f1s), MathBoolConstant<bool, true, true>>::value,
                  "Wrong result type!");
    static_assert(std::is_same<decltype(f2s), MathBoolConstant<bool, true, true>>::value,
                  "Wrong result type!");
    static_assert(std::is_same<decltype(f3s), MathBoolConstant<bool, false, false>>::value,
                  "Wrong result type!");
    static_assert(!decltype(f4s)::lb, "Wrong resulting bound!");
    static_assert(decltype(f4s)::ub, "Wrong resulting bound!");

    REQUIRE(f1s(15.5, 8.5));
    REQUIRE(f2s(16.0, 8.0));
    REQUIRE(!f3s(15.0, 7.0));
    REQUIRE(!f4s(14.1, 7.0));
    REQUIRE(f4s(14.0, 7.0));
}

TEST_CASE("[ivarp][constraint refactoring] Simplify and bound - if then else") {
    auto swapsgn = if_then_else(args::x0 < 0_Z, -args::x1, args::x1);
    using F = decltype(swapsgn);

    REQUIRE(swapsgn(-1., -2.) == 2.);
    REQUIRE(swapsgn(0., -2.) == -2.);
    REQUIRE(swapsgn(1., -5.) == -5.);
    REQUIRE(swapsgn(IDouble{-1.,1.}, 5.).same(IDouble{-5.,5.}));

    using BPos = Tuple<ExpressionBounds<0,1>,
                       ExpressionBounds<fixed_point_bounds::int_to_fp(1), fixed_point_bounds::int_to_fp(2)>>;
    using BNeg = Tuple<ExpressionBounds<-2,-1>,
                       ExpressionBounds<fixed_point_bounds::int_to_fp(1), fixed_point_bounds::int_to_fp(2)>>;
    using BMix = Tuple<ExpressionBounds<-1,0>,
                       ExpressionBounds<fixed_point_bounds::int_to_fp(1), fixed_point_bounds::int_to_fp(2)>>;

    auto bp = bound_and_simplify<BPos>(F(swapsgn));
    auto bn = bound_and_simplify<BNeg>(F(swapsgn));
    auto bm = bound_and_simplify<BMix>(F(swapsgn));
    using BP = decltype(bp);
    using BN = decltype(bn);
    using BM = decltype(bm);
    static_assert(BP::lb == fixed_point_bounds::int_to_fp(1), "Wrong LB!");
    static_assert(BP::ub == fixed_point_bounds::int_to_fp(2), "Wrong UB!");
    static_assert(BN::ub == -fixed_point_bounds::int_to_fp(1), "Wrong UB!");
    static_assert(BN::lb == -fixed_point_bounds::int_to_fp(2), "Wrong LB!");
    static_assert(BM::lb == -fixed_point_bounds::int_to_fp(2), "Wrong LB!");
    static_assert(BM::ub == fixed_point_bounds::int_to_fp(2), "Wrong UB!");
}

TEST_CASE("[ivarp][constraint refactoring] Simplify and bound - non-bound evaluating function") {
    auto fn = [] (const auto& /*ctx*/, auto && x, auto && y) { return x + y; };
    auto cfn = custom_function_context_as_arg(fn, args::x0, args::x1);
    using CFBounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(33), fixed_point_bounds::int_to_fp(34)>,
                           ExpressionBounds<fixed_point_bounds::int_to_fp(5), fixed_point_bounds::int_to_fp(8)>>;
    auto s = bound_and_simplify<CFBounds>(ivarp::move(cfn));
    using CFS = decltype(s);

    static_assert(!fixed_point_bounds::is_lb(CFS::lb), "Wrong lb!");
    static_assert(!fixed_point_bounds::is_ub(CFS::ub), "Wrong ub!");
    REQUIRE(s(33.0, 5.0) == 38.0);
    REQUIRE(s(33.5, 6.5) == 40.0);
    REQUIRE(s(34.0, 8.0) == 42.0);
}
