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
    template<typename Index_> struct MathArg : MathExpressionBase<MathArg<Index_>> {
        static constexpr unsigned index = Index_::value;
        static constexpr bool cuda_supported = true;
    };

    template<typename T> struct IsMathArgImpl : std::false_type {};
    template<typename T> struct IsMathArgImpl<MathArg<T>> : std::true_type {};
    template<typename T> using IsMathArg = IsMathArgImpl<std::decay_t<T>>;
    template<std::size_t Index> using MathArgFromIndex =
        MathArg<std::integral_constant<unsigned, Index>>;

    namespace args {
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<0> x0;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<1> x1;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<2> x2;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<3> x3;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<4> x4;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<5> x5;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<6> x6;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<7> x7;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<8> x8;
        static IVARP_CUDA_DEVICE_OR_CONSTEXPR MathArgFromIndex<9> x9;
    }
}
