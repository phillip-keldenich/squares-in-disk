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
    /// Tag for ^ (which is a binary-only operator).
    struct MathPredXor : MathPredJunctor {
        static constexpr PrintOp print_operator = PrintOp::XOR;

        template<typename Context, typename V1, typename V2>
            static inline IVARP_HD impl::PredicateEvalResultType<Context> 
                eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 ^ v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
                struct Eval
            {
            private:
                static constexpr bool known = (B1::lb == B1::ub) && (B2::lb == B2::ub);

            public:
                static constexpr bool lb = known ? (B1::lb ^ B2::lb) : false;
                static constexpr bool ub = known ? lb : true;
            };
        };

        /// Simplify checks bounds 
        struct BoundAndSimplify {
            template<typename Old, typename B1, typename B2,
                     std::enable_if_t<!B1::lb && !B1::ub, int> = 0>
                static inline decltype(auto) bound_and_simplify(Old&&, B1&&, B2&& arg2)
            {
                return ivarp::forward<B2>(arg2);
            }

            template<typename Old, typename B1, typename B2,
                std::enable_if_t<B1::lb && B1::ub, int> = 0>
                static inline decltype(auto) bound_and_simplify(Old&&, B1&&, B2&& arg2)
            {
                return !ivarp::forward<B2>(arg2);
            }

            template<typename Old, typename B1, typename B2,
                std::enable_if_t<B1::lb != B1::ub, int> = 0,
                std::enable_if_t<!B2::lb && !B2::ub, int> = 0>
                static inline decltype(auto) bound_and_simplify(Old&&, B1&& arg1, B2&&)
            {
                return ivarp::forward<B1>(arg1);
            }

            template<typename Old, typename B1, typename B2,
                std::enable_if_t<B1::lb != B1::ub, int> = 0,
                std::enable_if_t<B2::lb && B2::ub, int> = 0>
                static inline decltype(auto) bound_and_simplify(Old&&, B1&& arg1, B2&&)
            {
                return !ivarp::forward<B1>(arg1);
            }

            template<typename Old, typename B1, typename B2,
                std::enable_if_t<B1::lb != B1::ub, int> = 0,
                std::enable_if_t<B2::lb != B2::ub, int> = 0>
                static inline decltype(auto) bound_and_simplify(Old&&, B1&& arg1, B2&& arg2)
            {
                using Unwrapped = BinaryMathPred<MathPredXor, BareType<B1>, BareType<B2>>;
                return BoundedPredicate<Unwrapped, fixed_point_bounds::UnboundedPredicate>{
                    Unwrapped{ ivarp::forward<B1>(arg1), ivarp::forward<B2>(arg2) }
                };
            }
        };
    };

    /// XOR operator (binary only).
    template<typename LHS, typename RHS,
             std::enable_if_t<(IsMathPred<LHS>::value || IsMathPred<RHS>::value) &&
                              (IsMathPred<LHS>::value || IsBoolean<LHS>::value) &&
                              (IsMathPred<RHS>::value || IsBoolean<RHS>::value),int> = 0>
    static inline BinaryMathPred<MathPredXor,EnsurePred<LHS>,EnsurePred<RHS>> operator^(LHS&& l, RHS&& r) {
        return {std::forward<LHS>(l), std::forward<RHS>(r)};
    }

    /// Tag for <=.
    struct BinaryMathPredLEQ : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::LEQ;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::LEQ;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::GEQ;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 <= v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 <= v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
                struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::LE || order == Order::LT || order == Order::EQ);
                static constexpr bool ub = (order == Order::UNKNOWN || order == Order::LE || 
                                            order == Order::GE || order == Order::LT || order == Order::EQ);
            };
        };
    };

    /// Tag for <.
    struct BinaryMathPredLT : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::LT;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::LEQ;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::GEQ;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 < v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 < v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::LT);
                static constexpr bool ub = (order == Order::UNKNOWN || order == Order::LE || order == Order::LT);
            };
        };
    };

    /// Tag for >=.
    struct BinaryMathPredGEQ : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::GEQ;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::GEQ;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::LEQ;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 >= v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 >= v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::GE || order == Order::GT || order == Order::EQ);
                static constexpr bool ub = (order == Order::UNKNOWN || order == Order::GE ||
                                            order == Order::LE || order == Order::GT || order == Order::EQ);
            };
        };
    };

    /// Tag for >.
    struct BinaryMathPredGT : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::GT;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::GEQ;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::LEQ;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 > v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 > v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::GT);
                static constexpr bool ub = (order == Order::UNKNOWN || order == Order::GE || order == Order::GT);
            };
        };
    };

    /// Tag for ==.
    struct BinaryMathPredEQ : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::EQ;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::BOTH;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::BOTH;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 == v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 == v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::EQ);
                static constexpr bool ub = (order != Order::GT && order != Order::LT);
            };
        };
    };

    /// Tag for !=.
    struct BinaryMathPredNEQ : MathPredTerm {
        static constexpr PrintOp print_operator = PrintOp::NEQ;
        static constexpr BoundDirection lhs_bound_direction = BoundDirection::NONE;
        static constexpr BoundDirection rhs_bound_direction = BoundDirection::NONE;

        template<typename Context, typename V1, typename V2> static inline IVARP_HD
            EnableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2) noexcept
        {
            return v1 != v2;
        }

        template<typename Context, typename V1, typename V2> static inline IVARP_H
            DisableForCUDANT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> eval(const V1& v1, const V2& v2)
        {
            return v1 != v2;
        }

        struct EvalBounds {
            template<typename B1, typename B2>
            struct Eval
            {
            private:
                using Order = fixed_point_bounds::Order;
                static constexpr Order order = fixed_point_bounds::iv_order(B1::lb, B1::ub, B2::lb, B2::ub);

            public:
                static constexpr bool lb = (order == Order::GT || order == Order::LT);
                static constexpr bool ub = (order != Order::EQ);
            };
        };
    };

    /// Swap the sides of a binary relational operator; <= becomes >=, < becomes >, == and != remain the same.
    template<typename TagT> struct SwapSidesImpl;
    template<> struct SwapSidesImpl<BinaryMathPredEQ> { using Type = BinaryMathPredEQ; };
    template<> struct SwapSidesImpl<BinaryMathPredNEQ> { using Type = BinaryMathPredNEQ; };
    template<> struct SwapSidesImpl<BinaryMathPredLT> { using Type = BinaryMathPredGT; };
    template<> struct SwapSidesImpl<BinaryMathPredGT> { using Type = BinaryMathPredLT; };
    template<> struct SwapSidesImpl<BinaryMathPredLEQ> { using Type = BinaryMathPredGEQ; };
    template<> struct SwapSidesImpl<BinaryMathPredGEQ> { using Type = BinaryMathPredLEQ; };
    template<typename TagT> using SwapSides = typename SwapSidesImpl<TagT>::Type;

// Definition of symbolic comparison operators between MathExpressions.
// This is very repetitive code that is easier to maintain if there is only one copy of it
#define IVARP_DEFINE_MATH_TERM_2EXPR_OP(op, TagName)\
    template<typename LHS, typename RHS, std::enable_if_t<IsMathExpr<LHS>::value && IsMathExpr<RHS>::value, int> = 0>\
        static inline BinaryMathPred<BinaryMathPred##TagName, BareType<LHS>, BareType<RHS>>\
            operator op(LHS&& lhs, RHS&& rhs)\
    {\
        return {ivarp::forward<LHS>(lhs), ivarp::forward<RHS>(rhs)};\
    }

    IVARP_DEFINE_MATH_TERM_2EXPR_OP(<, LT)
    IVARP_DEFINE_MATH_TERM_2EXPR_OP(<=, LEQ)
    IVARP_DEFINE_MATH_TERM_2EXPR_OP(>, GT)
    IVARP_DEFINE_MATH_TERM_2EXPR_OP(>=, GEQ)
    IVARP_DEFINE_MATH_TERM_2EXPR_OP(==, EQ)
    IVARP_DEFINE_MATH_TERM_2EXPR_OP(!=, NEQ)
#undef IVARP_DEFINE_MATH_TERM_2EXPR_OP

// Definition of symbolic operators between MathExpressions and number types implicitly convertible to MathConstant.
#define IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(op, TagName)\
    template<typename LHS, typename RHS,\
             std::enable_if_t<IsMathExpr<LHS>::value && ImplicitConstantPromotion<RHS>::value, int> = 0>\
    static inline BinaryMathPred<BinaryMathPred##TagName, BareType<LHS>, NumberToConstant<RHS>>\
        operator op(LHS&& lhs, RHS&& rhs)\
    {\
        return {ivarp::forward<LHS>(lhs), NumberToConstant<RHS>{ivarp::forward<RHS>(rhs)}};\
    }\
    template<typename LHS, typename RHS,\
             std::enable_if_t<IsMathExpr<RHS>::value && ImplicitConstantPromotion<LHS>::value, int> = 0>\
    static inline BinaryMathPred<BinaryMathPred##TagName, NumberToConstant<LHS>, BareType<RHS>>\
        operator op(LHS&& lhs, RHS&& rhs)\
    {\
        return {NumberToConstant<LHS>(ivarp::forward<LHS>(lhs)), ivarp::forward<RHS>(rhs)};\
    }

    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(<, LT)
    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(<=, LEQ)
    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(>, GT)
    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(>=, GEQ)
    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(==, EQ)
    IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP(!=, NEQ)
#undef IVARP_DEFINE_MATH_TERM_EXPR_VAL_OP
}
