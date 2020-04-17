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
// Created by Phillip Keldenich on 19.11.19.
//

#pragma once

namespace ivarp {
    template<typename MathExprOrPred> using CanConstantFold =
        std::integral_constant<bool, (NumArgs<MathExprOrPred>::value == 0)>;

    template<typename MathExpr, std::int64_t LB, std::int64_t UB> struct ConstantFoldedExpr :
        MathExpressionBase<ConstantFoldedExpr<MathExpr, LB, UB>>
    {
        static_assert(CanConstantFold<MathExpr>::value, "ConstantFoldedExpr base depends on an argument!");

        using RationalPromoted = typename impl::IntervalPromotion<MathExpr, Rational>::Type;
        static constexpr bool is_exact = IsRational<RationalPromoted>::value;
        static constexpr bool cuda_supported = false;
        static constexpr std::int64_t lb = LB;
        static constexpr std::int64_t ub = UB;
        using RationalCtx = DefaultContextWithNumberType<RationalPromoted>;

        explicit ConstantFoldedExpr(const MathExpr& base) :
            base(base),
            irational(base.template evaluate<RationalCtx>()),
            ifloat(convert_number<IFloat>(irational)),
            idouble(convert_number<IDouble>(irational))
        {}

        IVARP_H ConstantFoldedExpr(const ConstantFoldedExpr& o) :
            base(o.base),
            irational(o.irational),
            ifloat(o.ifloat),
            idouble(o.idouble)
        {}

        MathExpr  base;
        RationalPromoted irational;
        IFloat    ifloat;
        IDouble   idouble;
    };
}
