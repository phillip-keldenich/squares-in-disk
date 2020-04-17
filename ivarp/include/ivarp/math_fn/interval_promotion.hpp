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
// Created by Phillip Keldenich on 05.10.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// A metafunction that detects whether for a given math expression and evaluation number type,
    /// interval promotion is necessary. float, double and interval types are never promoted;
    /// Rational is promoted if the expression involves anything that results in non-rational numbers, i.e.,
    /// interval constants or irrational functions.
    template<typename MathExpr, typename EvalNumberType> struct IntervalPromotion {
        using Type = EvalNumberType;
        static_assert(!std::is_same<EvalNumberType,Rational>::value,
                      "Wrong specialization chosen for IntervalPromotion of a Rational!");
    };

    template<typename MathExprOrPred> struct IntervalPromotion<MathExprOrPred, Rational> {
        static_assert(IsMathExprOrPred<MathExprOrPred>::value,
                      "IntervalPromotion used on non-expression/non-predicate type!");
        using Type = std::conditional_t<PreservesRationality<MathExprOrPred>::value &&
                                        !HasIntervalConstants<MathExprOrPred>::value, Rational, IRational>;
    };
}
}
