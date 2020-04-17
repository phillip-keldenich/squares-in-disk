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
// Created by Phillip Keldenich on 26.02.20.
//

/// @file see https://gitlab.ibr.cs.tu-bs.de/alg/ivarp/issues/8

#include "../test_util.hpp"
#include "ivarp/refactor_constraint_system.hpp"

namespace {
namespace issue8 {
    using namespace ivarp;
    using namespace ivarp::args;

    const auto T = (2_Z / sqrt(ensure_expr(5_Z))) * sqrt(maximum(0_Z, 1_Z - 0.2_X * square(x0))) - 0.8_X * x0;
    const auto T_inv = sqrt(maximum(1_Z - 0.25_X * square(x0), 0_Z)) - x0;

    using ArgBounds = Tuple<
        ExpressionBounds<295 * fixed_point_bounds::denom() / 1000, 13 * fixed_point_bounds::denom() / 10>, // s1: [0.295,1.3]
        ExpressionBounds<0, 13 * fixed_point_bounds::denom() / 10>, // all others: [0,1.3]
        ExpressionBounds<0, 13 * fixed_point_bounds::denom() / 10>,
        ExpressionBounds<0, 13 * fixed_point_bounds::denom() / 10>
    >;

    const auto z_part1 = sqrt(
        maximum(0_Z, 1_Z - 0.2_X * square(-(sqrt(maximum(0_Z, 1_Z - 0.25_X * square(x0))) - x0) + x1 + x2 + x3))
    );
    using P1Type = BareType<decltype(z_part1)>;
    const auto bs_z_part1 = ivarp::impl::BoundAndSimplify<P1Type, ArgBounds>::apply(P1Type(z_part1));
    using BSP1 = BareType<decltype(bs_z_part1)>;
    static_assert(BSP1::lb >= 0, "Missing/weak LB!");
    static_assert(BSP1::ub <= fixed_point_bounds::int_to_fp(1), "Missing/weak UB!");

    const auto unbounded_expr = ensure_expr(std::int64_t{6});
    using UnbType = BareType<decltype(unbounded_expr)>;
    static_assert(!fixed_point_bounds::is_lb(UnbType::lb), "Wrong LB!");
    static_assert(!fixed_point_bounds::is_ub(UnbType::ub), "Wrong UB!");

    const auto mult = 0.2_X * unbounded_expr;
    using MultType = BareType<decltype(mult)>;
    const auto bs_mult = ivarp::impl::BoundAndSimplify<MultType, ArgBounds>::apply(MultType(mult));
    static_assert(!fixed_point_bounds::is_lb(decltype(bs_mult)::lb), "Wrong LB!");
    static_assert(!fixed_point_bounds::is_ub(decltype(bs_mult)::ub), "Wrong UB!");

    const auto mult2 = -0.2_X * unbounded_expr;
    using MultType2 = BareType<decltype(mult2)>;
    const auto bs_mult2 = ivarp::impl::BoundAndSimplify<MultType2, ArgBounds>::apply(MultType2(mult2));
    static_assert(!fixed_point_bounds::is_lb(decltype(bs_mult2)::lb), "Wrong LB!");
    static_assert(!fixed_point_bounds::is_ub(decltype(bs_mult2)::ub), "Wrong UB!");

    const auto mult3 = unbounded_expr * 0.2_X;
    using MultType3 = BareType<decltype(mult3)>;
    const auto bs_mult3 = ivarp::impl::BoundAndSimplify<MultType3, ArgBounds>::apply(MultType3(mult3));
    static_assert(!fixed_point_bounds::is_lb(decltype(bs_mult3)::lb), "Wrong LB!");
    static_assert(!fixed_point_bounds::is_ub(decltype(bs_mult3)::ub), "Wrong UB!");

    const auto mult4 = unbounded_expr * -0.2_X;
    using MultType4 = BareType<decltype(mult4)>;
    const auto bs_mult4 = ivarp::impl::BoundAndSimplify<MultType4, ArgBounds>::apply(MultType4(mult4));
    static_assert(!fixed_point_bounds::is_lb(decltype(bs_mult4)::lb), "Wrong LB!");
    static_assert(!fixed_point_bounds::is_ub(decltype(bs_mult4)::ub), "Wrong UB!");
}
}
