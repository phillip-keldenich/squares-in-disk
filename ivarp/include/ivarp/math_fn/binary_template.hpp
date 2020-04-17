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
// Created by Phillip Keldenich on 26.10.19.
//

#pragma once

namespace ivarp {
    template<typename Tag_, typename A1, typename A2> struct MathBinary : MathExpressionBase<MathBinary<Tag_,A1,A2>> {
        using Tag = Tag_;
        using Arg1 = A1;
        using Arg2 = A2;

        static constexpr bool cuda_supported = Arg1::cuda_supported && Arg2::cuda_supported;

        static_assert(IsMathExpr<Arg1>::value, "First argument of binary expression is not a MathExpression!");
        static_assert(IsMathExpr<Arg2>::value, "Second argument of binary expression is not a MathExpression!");

        template<typename AA1, typename AA2>
        explicit constexpr MathBinary(AA1&& arg1, AA2&& arg2) :
                arg1(std::forward<AA1>(arg1)),
                arg2(std::forward<AA2>(arg2))
        {}

        IVARP_DEFAULT_CM(MathBinary);

        Arg1 arg1;
        Arg2 arg2;
    };
}
