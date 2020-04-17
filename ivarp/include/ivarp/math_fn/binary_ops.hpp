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
// Created by Phillip Keldenich on 2019-10-01.
//

#pragma once

namespace ivarp {
    struct MathOperatorTagAdd {
        static constexpr PrintOp print_operator = PrintOp::PLUS;

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::fp_add_rd(B1::lb, B2::lb);
                static constexpr std::int64_t ub = fixed_point_bounds::fp_add_ru(B1::ub, B2::ub);
            };
        };

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
            static inline NumberType
                eval(const NumberType& a1, const NumberType& a2) noexcept(AllowsCuda<NumberType>::value)
            {
                return a1 + a2;
            }
        )
    };

    struct MathOperatorTagSub {
        static constexpr PrintOp print_operator = PrintOp::MINUS;

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::fp_add_rd(B1::lb, -B2::ub);
                static constexpr std::int64_t ub = fixed_point_bounds::fp_add_ru(B1::ub, -B2::lb);
            };
        };

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
            static inline NumberType
                eval(const NumberType& a1, const NumberType& a2) noexcept(AllowsCuda<NumberType>::value)
            {
                return a1 - a2;
            }
        )
    };

    struct MathOperatorTagMul {
        static constexpr PrintOp print_operator = PrintOp::MUL;
        struct EvalBounds {
            template<typename B1, typename B2> struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::fp_iv_mul_lb(B1::lb, B1::ub, B2::lb, B2::ub);
                static constexpr std::int64_t ub = fixed_point_bounds::fp_iv_mul_ub(B1::lb, B1::ub, B2::lb, B2::ub);
            };
        };

        struct BoundedEval {
            /// Interval multiplication becomes cheaper if we know at least one sign;
            /// it may even becomes branchless if we know both and can exclude infinities.
            IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
                IVARP_TEMPLATE_PARAMS(typename Context, typename B1, typename B2, typename NumberType), NumberType,
                static inline NumberType eval(const NumberType& a1, const NumberType& a2) {
                    return BoundedMul<NumberType,B1,B2>::eval(a1, a2);
                }
            )
        };

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
            static inline NumberType
                eval(const NumberType& a1, const NumberType& a2) noexcept(AllowsCuda<NumberType>::value)
            {
                return a1 * a2;
            }
        )
    };

    struct MathOperatorTagDiv {
        static constexpr PrintOp print_operator = PrintOp::DIV;
        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::fp_iv_div_lb(B1::lb, B1::ub, B2::lb, B2::ub);
                static constexpr std::int64_t ub = fixed_point_bounds::fp_iv_div_ub(B1::lb, B1::ub, B2::lb, B2::ub);
            };
        };

        struct BoundedEval {
            /// Interval division becomes cheaper if we know signs in advance.
            IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
                IVARP_TEMPLATE_PARAMS(typename Context, typename B1, typename B2, typename NumberType), NumberType,
                static inline NumberType eval(const NumberType& a1, const NumberType& a2) {
                    return BoundedDiv<NumberType,B1,B2>::eval(a1, a2);
                }
            )
        };

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
            static inline NumberType
                eval(const NumberType& a1, const NumberType& a2) noexcept(AllowsCuda<NumberType>::value)
            {
                return a1 / a2;
            }
        )
    };

    template<typename A1, typename A2> using MathAdd = MathBinary<MathOperatorTagAdd, A1, A2>;
    template<typename A1, typename A2> using MathSub = MathBinary<MathOperatorTagSub, A1, A2>;
    template<typename A1, typename A2> using MathMul = MathBinary<MathOperatorTagMul, A1, A2>;
    template<typename A1, typename A2> using MathDiv = MathBinary<MathOperatorTagDiv, A1, A2>;

    /// Define operators for +,-,*,/ between two expressions or an expression and a constant that
    /// is implicitly convertible to a constant-expression.
#define IVARP_NS_SCOPE_DEFINE_BINARY_OP(op, Tag)\
    template<typename A1, typename A2,\
             std::enable_if_t<(IsMathExpr<A1>::value || ImplicitConstantPromotion<A1>::value) &&\
                              (IsMathExpr<A2>::value || ImplicitConstantPromotion<A2>::value) &&\
                              (IsMathExpr<A1>::value || IsMathExpr<A2>::value),int> = 0>\
        static inline auto operator op(A1&& a1, A2&& a2)\
    {\
        return MathBinary<Tag,EnsureExpr<A1>,EnsureExpr<A2>>{\
            ensure_expr(std::forward<A1>(a1)), ensure_expr(std::forward<A2>(a2))\
        };\
    }

    IVARP_NS_SCOPE_DEFINE_BINARY_OP(+, MathOperatorTagAdd)
    IVARP_NS_SCOPE_DEFINE_BINARY_OP(-, MathOperatorTagSub)
    IVARP_NS_SCOPE_DEFINE_BINARY_OP(*, MathOperatorTagMul)
    IVARP_NS_SCOPE_DEFINE_BINARY_OP(/, MathOperatorTagDiv)

#undef IVARP_NS_SCOPE_DEFINE_BINARY_OP
}
