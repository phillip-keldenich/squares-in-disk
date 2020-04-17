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
    /// Tag for if_then_else; implementation in predicate_eval.hpp due to sequencing semantics.
    /**
     * @brief Tag for if_then_else expressions.
     * Due to sequencing semantics and the need to evaluate predicates in order to evaluate an expression,
     * the evaluation is implemented in predicate_eval.hpp.
     */
    struct MathTernaryIfThenElse {
        static const char* name() noexcept {
            return "if_then_else";
        }

        struct EvalBounds {
            template<typename B1, typename B2, typename B3>
                struct Eval
            {
            private:
                using Bnd3 = std::conditional_t<B1::lb, B2, B3>;
                using Bnd2 = std::conditional_t<!B1::ub, B3, B2>;

            public:
                static constexpr std::int64_t lb = ivarp::min(Bnd2::lb, Bnd3::lb);
                static constexpr std::int64_t ub = ivarp::max(Bnd2::ub, Bnd3::ub);
            };
        };

        struct BoundAndSimplify {
            template<typename Old, typename BCond, typename BThen, typename BElse,
                     std::enable_if_t<BCond::lb && BCond::ub, int> = 0>
                static inline std::remove_reference_t<BThen> bound_and_simplify(Old&&, BCond&&, BThen&& then, BElse&&) noexcept
            {
                return ivarp::forward<BThen>(then);
            }

            template<typename Old, typename BCond, typename BThen, typename BElse,
                std::enable_if_t<!BCond::lb && !BCond::ub, int> = 0>
                static inline std::remove_reference_t<BElse> bound_and_simplify(Old&&, BCond&&, BThen&&, BElse&& else_) noexcept
            {
                return ivarp::forward<BElse>(else_);
            }

            template<typename Old, typename BCond, typename BThen, typename BElse,
                std::enable_if_t<!BCond::lb && BCond::ub, int> = 0>
                static inline auto bound_and_simplify(Old&&, BCond&& cond, BThen&& then, BElse&& else_) noexcept
            {
                using Inner = MathTernary<MathTernaryIfThenElse, BareType<BCond>, BareType<BThen>, BareType<BElse>>;

                return BoundedMathExpr<
                    Inner, ExpressionBounds<ivarp::min(BThen::lb,BElse::lb),ivarp::max(BThen::ub,BElse::ub)>
                >{Inner{ivarp::forward<BCond>(cond), ivarp::forward<BThen>(then), ivarp::forward<BElse>(else_)}};
            }
        };
    };

    /// Ternary if-then-else operator.
    template<typename Predicate, typename Then, typename Else,
             std::enable_if_t<IsPredOrConstant<Predicate>::value &&
                              IsExprOrConstant<Then>::value &&
                              IsExprOrConstant<Else>::value, int> = 0>
        static inline MathTernary<MathTernaryIfThenElse, EnsurePred<Predicate>, EnsureExpr<Then>, EnsureExpr<Else>>
            if_then_else(Predicate&& pred, Then&& then, Else&& else_)
    {
        return {
            ensure_pred(ivarp::forward<Predicate>(pred)),
            ensure_expr(ivarp::forward<Then>(then)),
            ensure_expr(ivarp::forward<Else>(else_))
        };
    }
}
