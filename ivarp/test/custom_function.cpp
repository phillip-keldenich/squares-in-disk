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
// Created by Phillip Keldenich on 22.11.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

using namespace ivarp;

namespace {
    struct CustomFunctor1 {
        template<typename Context, typename A1> auto eval(const A1& a1) const {
            return a1;
        }

        template<typename Context, typename A1, typename A2, typename... Args>
            auto eval(const A1& a1, const A2& a2, const Args&... args) const
        {
            return a1 + eval<Context>(a2, args...);
        }
    };

    const auto custom2 = [] (const auto& ctx, const auto& a1, const auto& a2) -> auto {
        using Context = std::decay_t<decltype(ctx)>;
        if(Context::irrational_precision != default_irrational_precision) {
            REQUIRE(false);
        }
        return a1 + a2;
    };

    const auto fn_custom1 = custom_function_context_as_template(CustomFunctor1{}, args::x0, args::x1, constant(0));
    const auto fn_custom2 = custom_function_context_as_arg(custom2, args::x0, args::x1);
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] Custom function evaluation", IT, IFloat, IDouble, IRational) {
    IT i1(1), i2(2);
    REQUIRE_SAME(fn_custom1(i1, i2), fn_custom2(i1, i2));
    REQUIRE_SAME(fn_custom1(i1,i2), IT(3));
    REQUIRE(definitely(fn_custom2(i1,i2) == 3));
}
