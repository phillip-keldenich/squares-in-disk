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
// Created by Phillip Keldenich on 14.12.19.
//

#pragma once

namespace ivarp {
    template<typename T, bool LB, bool UB> struct MathBoolConstant : MathPredBase<MathBoolConstant<T, LB, UB>> {
        static_assert(IsBoolean<T>::value, "The type of a MathBoolConstant value must be Boolean!");
        static constexpr bool cuda_supported = true;
        static constexpr bool lb = LB;
        static constexpr bool ub = UB;

        template<typename A, std::enable_if_t<!std::is_same<std::decay_t<A>, MathBoolConstant>::value, int> = 0>
            IVARP_HD explicit MathBoolConstant(A&& arg) noexcept :
                value(ivarp::forward<A>(arg))
        {}

        T value;
    };

    static_assert(IsMathPred<MathBoolConstant<bool, false, true>>::value, "A boolean constant is a predicate!");
    static_assert(IsMathPred<MathBoolConstant<IBool, false, true>>::value, "A boolean constant is a predicate!");
}
