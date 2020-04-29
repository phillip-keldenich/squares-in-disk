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
    /// Implementation of evaluation for boolean constants.
    template<typename Context, typename T, typename ArgArray, bool LB, bool UB>
        struct PredicateEvaluateImpl<Context, MathBoolConstant<T,LB,UB>, ArgArray, false>
    {
        using CalledType = MathBoolConstant<T,LB,UB>;
        static IVARP_HD PredicateEvalResultType<Context> eval(const CalledType& c, const ArgArray&) {
            return c.value;
        }
    };

    /// Implementation of evaluation for bounded predicates.
    template<typename Context, typename Child, typename BoundType, typename ArgArray>
        struct PredicateEvaluateImpl<Context, BoundedPredicate<Child, BoundType>, ArgArray, false>
    {
        using CalledType = BoundedPredicate<Child, BoundType>;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static PredicateEvalResultType<Context> eval(const CalledType& c, const ArgArray& a)
                noexcept(AllowsCUDA<NumberType>::value)
            {
                // Compile-time constant check; this should either be removed or used to eliminate
                // the child evaluation by any decent optimizer.
                if(BoundType::lb == BoundType::ub) {
                    return BoundType::lb;
                }
                return PredicateEvaluateImpl<Context, Child, ArgArray>::eval(c.child, a);
            }
        )
    };
}
}
