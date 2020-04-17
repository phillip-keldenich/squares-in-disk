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
// Created by Phillip Keldenich on 21.11.19.
//

#include "ivarp/math_fn.hpp"

/// Compile-time only tests (via static_assert).

namespace {
    const auto x0 = ivarp::args::x0;
    const auto x1 = ivarp::args::x1;
    const auto x2 = ivarp::args::x2;

    // test symbolic call on unary functions
    const auto fn_minus = -x0;
    const auto fn_minus_sc_res = fn_minus(x1);
    const auto fn_minus_sc_res_expected = -x1;
    using FnMinusScRes = std::decay_t<decltype(fn_minus_sc_res)>;
    using FnMinusScResEx = std::decay_t<decltype(fn_minus_sc_res_expected)>;
    static_assert(std::is_same<FnMinusScRes, FnMinusScResEx>::value, "Incorrect result!");

    // test symbolic call on combination of binary functions
    const auto fn_binary_test = x0 * x1 + x1 / x2 - x2 / x0;
    const auto fn_binary_sc_res = fn_binary_test(ivarp::constant(2), x0, x1);
    const auto fn_binary_sc_res_expected = 2 * x0 + x0 / x1 - x1 / 2;
    using FnBinaryScRes = std::decay_t<decltype(fn_binary_sc_res)>;
    using FnBinaryScResEx = std::decay_t<decltype(fn_binary_sc_res_expected)>;
    static_assert(std::is_same<FnBinaryScRes,FnBinaryScResEx>::value, "Incorrect result!");

    // test symbolic call on unary predicate
    const auto pred_not = !(x0 <= x1);
    const auto pred_not_sc_res = pred_not(x1, -x0);
    const auto pred_not_sc_res_expected = !(x1 <= -x0);
    using PredNotScRes = std::decay_t<decltype(pred_not_sc_res)>;
    using PredNotScResEx = std::decay_t<decltype(pred_not_sc_res_expected)>;
    static_assert(std::is_same<PredNotScRes, PredNotScResEx>::value, "Incorrect result!");

    // test symbolic call on binary predicates
    const auto pred_leq = (x0 <= 11 * x1);
    const auto pred_leq_sc_res = pred_leq(x0 / 3, x0 - 5);
    const auto pred_leq_sc_res_expected = (x0 / 3 <= 11 * (x0 - 5));
    using PredLeqScRes = std::decay_t<decltype(pred_leq_sc_res)>;
    using PredLeqScResEx = std::decay_t<decltype(pred_leq_sc_res_expected)>;
    static_assert(std::is_same<PredLeqScRes, PredLeqScResEx>::value, "Incorrect result!");

    // test symbolic call on n-ary predicates
    const auto pred_or = (x0 <= 2 * x1 || x1 <= x2 || x0 >= x1 + x2 || x0 + x2 <= x1);
    const auto pred_or_sc_res = pred_or(x1, x2, x0);
    const auto pred_or_sc_res_expected = (x1 <= 2 * x2 || x2 <= x0 || x1 >= x2 + x0 || x1 + x0 <= x2);
    using PredOrScRes = std::decay_t<decltype(pred_or_sc_res)>;
    using PredOrScResEx = std::decay_t<decltype(pred_or_sc_res_expected)>;
    static_assert(std::is_same<PredOrScRes, PredOrScResEx>::value, "Incorrect result!");

    // test symbolic call on n-ary functions
    const auto fn_nary_test = ivarp::maximum(x0, x1, 2*x2, x2);
    const auto fn_nary_sc_res = fn_nary_test(2*x0, 2*x1, x2);
    const auto fn_nary_sc_res_expected = ivarp::maximum(2*x0, 2*x1, 2*x2, x2);
    using FnNaryScRes = std::decay_t<decltype(fn_nary_sc_res)>;
    using FnNaryScResEx = std::decay_t<decltype(fn_nary_sc_res_expected)>;
    static_assert(std::is_same<FnNaryScRes,FnNaryScResEx>::value, "Incorrect result!");

    // test symbolic call on if_then_else
    const auto ite_test = ivarp::if_then_else(x0 <= x2, 2 * x0, x1);
    const auto ite_sc_res = ite_test(2 / x0, x1, 1 / x1);
    const auto ite_sc_res_expected = ivarp::if_then_else(2/x0 <= 1/x1, 2 * (2 / x0), x1);
    using IteScRes = std::decay_t<decltype(fn_nary_sc_res)>;
    using IteScResEx = std::decay_t<decltype(fn_nary_sc_res_expected)>;
    static_assert(std::is_same<IteScRes, IteScResEx>::value, "Incorrect result!");

    // test symbolic call on custom functions
    struct CustomTestFn {
        template<typename Context, typename A1, typename A2> auto operator()(const A1& a1, const A2& a2) const {
            return a1 - a2;
        }
    };
    const auto cfn_test = ivarp::custom_function_context_as_template(CustomTestFn{}, x0 + x1, x2);
    static_assert(ivarp::NumArgs<std::decay_t<decltype(cfn_test)>>::value == 3, "Incorrect number of arguments!");
    const auto cfn_sc_res = cfn_test(x0, x0, x0);
    const auto cfn_sc_res_expected = ivarp::custom_function_context_as_template(CustomTestFn{}, x0 + x0, x0);
    static_assert(ivarp::NumArgs<std::decay_t<decltype(cfn_sc_res)>>::value == 1, "Incorrect number of arguments!");
    static_assert(std::is_same<decltype(cfn_sc_res), decltype(cfn_sc_res_expected)>::value, "Incorrect result!");
}
