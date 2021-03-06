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
// Created by Phillip Keldenich on 18.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename T, std::int64_t LB, std::int64_t UB, typename ArgBounds> struct BoundAndSimplify<MathConstant<T,LB,UB>, ArgBounds, void> {
        using OldType = MathConstant<T,LB,UB>;
        static inline IVARP_H auto apply(OldType&& old) noexcept {
            return static_cast<OldType&&>(old);
        }
    };
    template<typename T, bool LB, bool UB, typename ArgBounds> struct BoundAndSimplify<MathBoolConstant<T,LB,UB>, ArgBounds, void> {
        using OldType = MathBoolConstant<T,LB,UB>;
        static inline IVARP_H auto apply(OldType&& old) noexcept {
            return static_cast<OldType&&>(old);
        }
    };
    template<std::int64_t LB, std::int64_t UB, typename ArgBounds> struct BoundAndSimplify<MathCUDAConstant<LB,UB>, ArgBounds, void> {
        using OldType = MathCUDAConstant<LB,UB>;
        static inline IVARP_H auto apply(OldType&& old) noexcept {
            return static_cast<OldType&&>(old);
        }
    };
    template<typename Expr, std::int64_t LB, std::int64_t UB, typename ArgBounds> struct BoundAndSimplify<ConstantFoldedExpr<Expr,LB,UB>, ArgBounds, void> {
        using OldType = ConstantFoldedExpr<Expr, LB,UB>;
        static inline IVARP_H auto apply(OldType&& old) noexcept {
            return static_cast<OldType&&>(old);
        }
    };
    template<typename Pred, bool LB, bool UB, typename ArgBounds> struct BoundAndSimplify<ConstantFoldedPred<Pred,LB,UB>, ArgBounds, void> {
        using OldType = ConstantFoldedPred<Pred,LB,UB>;
        static inline IVARP_H auto apply(OldType&& old) noexcept {
            return static_cast<OldType&&>(old);
        }
    };
}
}
