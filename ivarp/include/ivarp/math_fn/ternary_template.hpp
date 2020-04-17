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
// Created by Phillip Keldenich on 30.10.19.
//

#pragma once

namespace ivarp {
    template<typename Tag_, typename A1, typename A2, typename A3>
        struct MathTernary : MathExpressionBase<MathTernary<Tag_,A1,A2,A3>>
    {
        using Tag = Tag_;
        using Arg1 = A1;
        using Arg2 = A2;
        using Arg3 = A3;

        static constexpr bool cuda_supported = Arg1::cuda_supported && Arg2::cuda_supported && Arg3::cuda_supported;

        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;

        template<typename AA1, typename AA2, typename AA3>
            MathTernary(AA1&& a1, AA2&& a2, AA3&& a3) :
                arg1(std::forward<AA1>(a1)),
                arg2(std::forward<AA2>(a2)),
                arg3(std::forward<AA3>(a3))
        {}

        IVARP_DEFAULT_CM(MathTernary);
    };
}
