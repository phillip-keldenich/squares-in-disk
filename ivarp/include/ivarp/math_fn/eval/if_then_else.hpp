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

#pragma once

namespace ivarp {
namespace impl {
    /// Implementation of if_then_else (it needs to evaluate a predicate in order to evaluate a function).
    template<typename Context, typename A1, typename A2, typename A3, typename ArgArray>
        struct EvaluateImpl<Context, MathTernary<MathTernaryIfThenElse, A1, A2, A3>, ArgArray>
    {
        using CalledType = MathTernary<MathTernaryIfThenElse, A1, A2, A3>;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static NumberType
                eval(const CalledType& c, const ArgArray& args) noexcept(AllowsCuda<NumberType>::value)
            {
                PredicateEvalResultType<Context> pred_result =
                    PredicateEvaluateImpl<Context, typename CalledType::Arg1, ArgArray>::eval(c.arg1, args);
                return do_eval(c, pred_result, args);
            }
        )

    private:
        // version for bool (concrete numbers, no join!)
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(
                typename BoolType,
                std::enable_if_t<std::is_same<BoolType,bool>::value,int> = 0
            ), NumberType,
            static NumberType do_eval(const CalledType& c, BoolType pred_result, const ArgArray& args)
            {
                if(pred_result) {
                    return EvaluateImpl<Context, typename CalledType::Arg2, ArgArray>::eval(c.arg2, args);
                } else {
                    return EvaluateImpl<Context, typename CalledType::Arg3, ArgArray>::eval(c.arg3, args);
                }
            }
        )

        // version for intervals (IBool)
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(
                typename BoolType,
                std::enable_if_t<std::is_same<BoolType,IBool>::value,int> = 0
            ), NumberType,
            static NumberType do_eval(const CalledType& c, const BoolType& pred_result, const ArgArray& args)
            {
                if(definitely(pred_result)) {
                    return EvaluateImpl<Context, typename CalledType::Arg2, ArgArray>::eval(c.arg2, args);
                } else if(!possibly(pred_result)) {
                    return EvaluateImpl<Context, typename CalledType::Arg3, ArgArray>::eval(c.arg3, args);
                } else {
                    auto r = EvaluateImpl<Context, typename CalledType::Arg2, ArgArray>::eval(c.arg2, args);
                    r.do_join(EvaluateImpl<Context, typename CalledType::Arg3, ArgArray>::eval(c.arg3, args));
                    return r;
                }
            }
        )
    };
}
}
