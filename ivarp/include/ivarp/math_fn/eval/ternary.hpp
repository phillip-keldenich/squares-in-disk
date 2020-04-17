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
    /// Handling ternary operators (except for if_then_else, which is implemented in predicate_eval).
    template<typename Context, typename ArgArray, typename Tag, typename A1, typename A2, typename A3>
        struct EvaluateImpl<Context, MathTernary<Tag,A1,A2,A3>, ArgArray>
    {
        static_assert(!std::is_same<Tag, MathTernaryIfThenElse>::value, "Wrong specialization chosen!");

        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline NumberType eval(const MathTernary<Tag,A1,A2,A3>& b, const ArgArray& a)
                noexcept(AllowsCuda<NumberType>::value)
            {
                return invoke_tag<Tag,Context>(a, b.arg1, b.arg2, b.arg3);
            }
        )
    };
}
}
