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
// Created by Phillip Keldenich on 23.01.20.
//

#include "ivarp/refactor_constraint_system.hpp"
#include "test_util.hpp"

using namespace ivarp;
using namespace ivarp::fixed_point_bounds;

namespace {
    const auto x = args::x0;
    const auto y = args::x1;

    TEST_CASE("[ivarp][refactor_constraint_system] Rewrite to bounds - Relational") {
        const auto f1 = (x <= 0);
        const auto f2 = (x >= 2);
        const auto c2 = f2.arg2;
        using C2T = BareType<decltype(c2)>;

        constexpr int minint = std::numeric_limits<int>::min();
        constexpr int maxint = std::numeric_limits<int>::max();
        static_assert(std::is_same<C2T, MathConstant<int, int_to_fp(minint), int_to_fp(maxint)>>::value, "Wrong RHS!");
        REQUIRE_SAME(c2(IDouble{-1,1}), IDouble(2,2,false));

        require_printable_same(f1, "x0 <= 0");
        require_printable_same(f2, "x0 >= 2");

        using XBounds = Tuple<ExpressionBounds<fixed_point_bounds::int_to_fp(-2), fixed_point_bounds::int_to_fp(2)>>;
        const auto res_bounds1 = impl::rewrite_to_bound<0, XBounds>(f1);
        const auto& rb1 = ivarp::template get<0>(res_bounds1);
        const auto res_bounds2 = impl::rewrite_to_bound<0, XBounds>(f2);
        const auto& rb2 = ivarp::template get<0>(res_bounds2);
        REQUIRE(rb1.get_direction() == BoundDirection::LEQ);
        REQUIRE(rb2.get_direction() == BoundDirection::GEQ);
        REQUIRE_SAME(rb1.bound(IDouble{-1,1}), IDouble(0,0,false));
        REQUIRE_SAME(rb2.bound(IDouble{-1,1}), IDouble(2,2,false));
    }

    TEST_CASE("[ivarp][refactor_constraint_system] Check successful rewrite") {
        const auto fn1 = (x <= 0);
        const auto fn2 = (y >= x);
        const auto fn3 = known_bounds<PredicateBounds<true,true>>(x <= 10);

        using C1 = BareType<decltype(fn1)>;
        using C2 = BareType<decltype(fn2)>;
        using C3 = BareType<decltype(fn3)>;

        using ArgBounds = Tuple<ExpressionBounds<-int_to_fp(5), int_to_fp(5)>,
                                ExpressionBounds<-int_to_fp(8), int_to_fp(8)>>;

        static_assert(impl::SuccessfulCTRewrite<0, ArgBounds, C1>::value, "Wrong check!");
        static_assert(!impl::SuccessfulCTRewrite<1, ArgBounds, C1>::value, "Wrong check!");
        static_assert(impl::SuccessfulCTRewrite<0, ArgBounds, C2>::value, "Wrong check!");
        static_assert(impl::SuccessfulCTRewrite<1, ArgBounds, C2>::value, "Wrong check!");
        static_assert(!impl::SuccessfulCTRewrite<2, ArgBounds, C2>::value, "Wrong check!");
        static_assert(impl::SuccessfulCTRewrite<0, ArgBounds, C3>::value, "Wrong check!");
        static_assert(!impl::SuccessfulCTRewrite<1, ArgBounds, C3>::value, "Wrong check!");
    }

    TEST_CASE("[ivarp][refactor_constraint_system] Check runtime constraint table") {
        auto custom_unrewritable_fn = [] (const auto& /*ctx*/, const auto& arg1, const auto& arg2) -> auto {
            return arg1 * arg2;
        };
        const auto cf = custom_function_context_as_arg(custom_unrewritable_fn, x, y);
        const auto fn1 = (x <= 0);
        const auto fn2 = (x >= -5);
        const auto fn3 = (y <= 2);
        const auto fn4 = (y >= x);
        const auto fn5 = (cf <= 9);

        require_printable_same(fn5, "custom0(x0, x1) <= 9");
        require_printable_same(fn5, "custom0(x0, x1) <= 9");

        using C1 = BareType<decltype(fn1)>;
        using C2 = BareType<decltype(fn2)>;
        using C3 = BareType<decltype(fn3)>;
        using C4 = BareType<decltype(fn4)>;
        using C5 = BareType<decltype(fn5)>;
        using Constraints = Tuple<C1,C2,C3,C4,C5>;
        static_assert(Constraints::size == 5, "Wrong tuple size!");
        using ArgBounds = Tuple<ExpressionBounds<-int_to_fp(5), int_to_fp(0)>,
                                ExpressionBounds<-int_to_fp(5), int_to_fp(2)>,
                                ExpressionBounds<-int_to_fp(15), int_to_fp(20)>>;

        const auto rct = make_runtime_constraint_table<ArgBounds>(ivarp::make_tuple(fn1, fn2, fn3, fn4, fn5));
        using RCT = BareType<decltype(rct)>;
        using RCT1 = RCT::At<0>;
        using RCT2 = RCT::At<1>;
        using RCT3 = RCT::At<2>;
        static_assert(RCT1::size == 0, "Wrong number of unrewritten constraints!");
        static_assert(RCT2::size == 1, "Wrong number of unrewritten constraints!");
        static_assert(std::is_same<RCT2::At<0>,C5>::value, "Invalid unrewritten constraints!");
        static_assert(std::is_same<RCT3, Constraints>::value, "Invalid unrewritten constraints!");
    }
}
