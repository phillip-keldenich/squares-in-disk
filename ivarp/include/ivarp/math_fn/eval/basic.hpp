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
    /// Handling constants.
    template<typename Context, typename ArgArray, typename T, std::int64_t LB, std::int64_t UB>
        struct EvaluateImpl<Context, MathConstant<T,LB,UB>, ArgArray>
    {
        using CNT = typename Context::NumberType;

        IVARP_SUPPRESS_HD
        IVARP_HD_OVERLOAD_ON_CUDA_NT(CNT,
            static inline CNT eval(const MathConstant<T,LB,UB>& c, const ArgArray&)
                noexcept(AllowsCUDA<CNT>::value)
            {
                return c.template as<CNT>();
            }
        );
    };

    /// Handling CUDA-compatible constants.
    template<typename Context, typename ArgArray, std::int64_t LB, std::int64_t UB>
        struct EvaluateImpl<Context, MathCUDAConstant<LB,UB>, ArgArray>
    {
        static inline IVARP_HD typename Context::NumberType
            eval(const MathCUDAConstant<LB,UB>& c, const ArgArray&) noexcept
        {
            return c.template as<typename Context::NumberType>();
        }
    };

    /// Evaluating parameters.
    template<typename Context, typename ArgArray, typename IT>
        struct EvaluateImpl<Context, MathArg<IT>, ArgArray>
    {
        static constexpr unsigned index = IT::value;

        using NumberType = typename Context::NumberType;
        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline NumberType eval(const MathArg<IT>&, const ArgArray& a)
                noexcept(AllowsCUDA<NumberType>::value)
            {
                return a[index];
            }
        )
    };

    /// Handle bounded expressions.
    template<typename Context, typename ArgArray, typename Child, typename Bounds>
        struct EvaluateImpl<Context, BoundedMathExpr<Child, Bounds>, ArgArray>
    {
        using CalledType = BoundedMathExpr<Child, Bounds>;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline IVARP_HD NumberType eval(const CalledType& c, const ArgArray& a)
                noexcept(AllowsCUDA<NumberType>::value)
            {
                return EvaluateImpl<Context, Child, ArgArray>::eval(c.child, a);
            }
        )
    };
}
}
