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
    /// Tag for ! (logical negation).
    struct UnaryMathPredNotTag : MathPredJunctor {
        static constexpr PrintOp print_operator = PrintOp::LOG_NEG;

        struct EvalBounds {
            template<typename B1>
            struct Eval {
                static constexpr bool lb = !B1::ub;
                static constexpr bool ub = !B1::lb;
            };
        };

        template<typename Context, typename ValueType>
        IVARP_HD static impl::PredicateEvalResultType<Context> eval(const ValueType& v) noexcept
        {
            return !v;
        }
    };

    /// Tag for ~ (undefinedness check).
    struct UnaryMathPredUndefTag : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::TILDE;

        template<typename Context, typename ValueType>
        static std::enable_if_t<IsIntervalType<ValueType>::value, impl::PredicateEvalResultType<Context>>
        IVARP_HD eval(const ValueType& v) noexcept
        {
            return impl::PredicateEvalResult<Context>{false, v.possibly_undefined()};
        }

        template<typename Context, typename ValueType>
        static std::enable_if_t<!IsIntervalType<ValueType>::value, impl::PredicateEvalResultType<Context>>
        IVARP_HD eval(const ValueType& v) noexcept
        {
            return impl::PredicateEvalResultType<Context>{v != v};
        }
    };

    template<typename A> using UnaryMathNot = UnaryMathPred<UnaryMathPredNotTag, std::decay_t<A>>;
    template<typename A> using UnaryMathUndef = UnaryMathPred<UnaryMathPredUndefTag, std::decay_t<A>>;

    /// Negation operator.
    template<typename PredType> static inline
        std::enable_if_t<IsMathPred<PredType>::value, UnaryMathNot<PredType>>
            operator!(PredType&& pred)
    {
        return UnaryMathNot<PredType>{std::forward<PredType>(pred)};
    }

    /// Undefinedness check operator.
    template<typename MathExpr> static inline
        std::enable_if_t<IsMathExpr<MathExpr>::value, UnaryMathUndef<MathExpr>>
            operator~(MathExpr&& expr)
    {
        return {std::forward<MathExpr>(expr)};
    }
}
