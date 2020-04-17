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
// Created by Phillip Keldenich on 19.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename Context, typename Base, bool LB, bool UB, typename ArgArray>
        struct PredicateEvaluateImpl<Context, ConstantFoldedPred<Base,LB,UB>, ArgArray, false>
    {
    private:
        using CalledType = ConstantFoldedPred<Base,LB,UB>;
        using NumberType = typename Context::NumberType;
        using ResultType = PredicateEvalResultType<Context>;
        using BoolTag = std::true_type;
        using IBoolTag = std::false_type;
        using ResultTypeTag = std::integral_constant<bool, std::is_same<ResultType, bool>::value>;
        using ValueType = typename CalledType::ValueType;

        static constexpr bool is_rational = IsRational<NumberType>::value || std::is_same<NumberType, IRational>::value;

        using IsSufficientlyPreciseTag = std::integral_constant<bool,
            !is_rational || Context::irrational_precision <= default_irrational_precision ||
            std::is_same<ValueType,bool>::value
        >;

        static bool as_result_type(bool b, BoolTag) noexcept { return b; }
        static bool as_result_type(IBool i, BoolTag) noexcept { return i.definitely(); }
        static IBool as_result_type(bool b, IBoolTag) noexcept { return {b}; }
        static IBool as_result_type(const IBool& i, IBoolTag) noexcept { return i; }

        static ResultType do_eval(const CalledType& c, const ArgArray&, std::true_type /*suff_precise*/) noexcept {
            return as_result_type(c.value, ResultTypeTag{});
        }

        static ResultType do_eval(const CalledType& c, const ArgArray& a, std::false_type /*suff_precise*/) {
            return PredicateEvaluateImpl<Context, Base, ArgArray>::eval(c,a);
        }

    public:
        static ResultType eval(const CalledType& c, const ArgArray& a) {
            return do_eval(c, a, IsSufficientlyPreciseTag{});
        }
    };
}
}
