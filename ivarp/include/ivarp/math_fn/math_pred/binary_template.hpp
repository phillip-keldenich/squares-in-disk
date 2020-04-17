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
// Created by Phillip Keldenich on 29.10.19.
//

#pragma once

namespace ivarp {
    /// Binary predicate template.
    template<typename Tag_, typename Arg1_, typename Arg2_> struct BinaryMathPred :
            MathPredBase<BinaryMathPred<Tag_,Arg1_,Arg2_>>
    {
        using Tag = Tag_;
        using Arg1 = Arg1_;
        using Arg2 = Arg2_;

        static constexpr bool cuda_supported = Arg1::cuda_supported && Arg2::cuda_supported;

        IVARP_DEFAULT_CM(BinaryMathPred);

        template<typename A1, typename A2> BinaryMathPred(A1&& a1, A2&& a2) :
                arg1(std::forward<A1>(a1)), arg2(std::forward<A2>(a2))
        {}

        Arg1 arg1;
        Arg2 arg2;
    };
}
