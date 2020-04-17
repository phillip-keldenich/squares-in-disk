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
// Created by Phillip Keldenich on 07.10.19.
//

#pragma once

namespace ivarp {
    /// The base class for all math expressions, i.e., functions on Numbers returning a Number.
    struct MathExpression {};

    /// The base class for all math predicates, i.e., functions on Numbers returning bool or IBool.
    struct MathPred {};

    /// Describe the rank of an operator for printing.
    enum class PrintRank : int {
        ID = 0,
        PARENTHESIS,
        UNARY,
        MULTIPLICATIVE,
        ADDITIVE,
        COMPARISON,
        AND, XOR, OR,
        UNKNOWN
    };

    static inline bool operator<(PrintRank p1, PrintRank p2) noexcept {
        return static_cast<int>(p1) < static_cast<int>(p2);
    }

    static inline bool operator>(PrintRank p1, PrintRank p2) noexcept {
        return static_cast<int>(p1) > static_cast<int>(p2);
    }

    static inline bool operator<=(PrintRank p1, PrintRank p2) noexcept {
        return static_cast<int>(p1) <= static_cast<int>(p2);
    }

    static inline bool operator>=(PrintRank p1, PrintRank p2) noexcept {
        return static_cast<int>(p1) >= static_cast<int>(p2);
    }

    /// Identify an operator for printing.
    enum class PrintOp : int {
        UNARY_MINUS, LOG_NEG, TILDE,
        PLUS, MINUS, MUL, DIV,
        LEQ, LT, GEQ, GT, EQ, NEQ, NONE,
        XOR, AND, OR
	};

    /// Base class for lookup tables for argument names.
    class ArgNameLookup {
    public:
        virtual ~ArgNameLookup() = default;
        virtual IVARP_H std::string arg_name(std::size_t arg_index) const = 0;
    };

    /// Forward declarations for all expression templates.
    template<typename MathExpr, std::int64_t LB, std::int64_t UB> struct ConstantFoldedExpr;
    template<typename Child, typename BoundsType> struct BoundedMathExpr;
    template<typename Tag_, typename A> struct MathUnary;
    template<typename Tag_, typename A1, typename A2> struct MathBinary;
    template<std::int64_t LB, std::int64_t UB> struct MathCudaConstant;
    template<typename Tag_, typename A1, typename A2, typename A3> struct MathTernary;
    template<typename Index_> struct MathArg;
    template<typename Tag_, typename... Args_> struct MathNAry;
    template<typename FunctorType_, typename... Args_> struct MathCustomFunction;

    /// Forward declarations for all predicate templates.
    template<typename MathPred, bool LB, bool UB> struct ConstantFoldedPred;
    template<typename Child, typename BoundsType> struct BoundedPredicate;
    template<typename Tag_, typename Arg> struct UnaryMathPred;
    template<typename T, bool LB, bool UB> struct MathBoolConstant;
    template<typename Tag_, typename Arg1_, typename Arg2_> struct BinaryMathPred;
    template<typename Tag_, typename... Args_> struct NAryMathPred;

    /// Removing bounds (from just the given type, not recursively).
    template<typename MathExprOrPred> struct StripBoundsImpl;
    template<typename MathExprOrPred> using StripBounds = typename StripBoundsImpl<MathExprOrPred>::Type;
    template<typename MathExprOrPred> static inline IVARP_H decltype(auto) strip_bounds(MathExprOrPred&& expr) {
        return StripBoundsImpl<BareType<MathExprOrPred>>::strip_bounds(ivarp::forward<MathExprOrPred>(expr));
    }

    /// Check for expressions/predicates.
    template<typename T> using IsMathExpr = std::is_convertible<BareType<T>, MathExpression>;
    template<typename T> using IsMathPred = std::is_convertible<BareType<T>, MathPred>;
    template<typename T> using IsMathExprOrPred = std::integral_constant<bool,
        IsMathExpr<T>::value || IsMathPred<T>::value>;
    template<typename T> using IsBoolOrPred = std::integral_constant<bool,
            IsBoolean<T>::value || IsMathPred<T>::value>;

namespace impl {
    /// Return the number of children the given expression/predicate type has.
    template<typename MathExprOrPred> struct NumChildren;

    /// Get the i-th child of the given expression/predicate type.
    template<typename MathExprOrPred, std::size_t I> struct ChildAt;
    template<typename MathExprOrPred, std::size_t I> using ChildAtType = typename ChildAt<MathExprOrPred,I>::Type;

    /// Check whether calling the given CalledType with the given Arguments is a symbolic call.
    template<typename CalledType, typename... ArgsPassed> struct IsSymbolicCall;

    /// Prepare the argument tuple for a symbolic (function or predicate) call; checks the number of arguments.
    template<typename CalledType, typename... ArgsPassed> struct SymbolicPrepareArgs;

    /// The resulting type of a symbolic call.
    template<typename CalledType, typename ArgTuple> struct SymbolicCallResult;
    template<typename CalledType, typename ArgTuple>
        using SymbolicCallResultType = typename SymbolicCallResult<CalledType, ArgTuple>::Type;

    /// A metafunction that checks whether a function or predicate preserves rationality of its input numbers, i.e.,
    /// whether the resulting number is always rational when all input arguments are rational.
    template<typename MathExprOrPred> struct PreservesRationality;

    /// A metafunction that checks whether a function or predicate uses interval constants.
    template<typename MathExprOrPred> struct HasIntervalConstants;

    /// The result of interval promotion for the given expression and evaluation number type;
    /// for instance, when evaluating a transcendental function such as sin or exp with Rational as number type,
    /// the evaluation number type is promoted to Interval<Rational>.
    template<typename MathExpr, typename EvalNumberType> struct IntervalPromotion;
    template<typename MathExpr, typename EvalNumberType> using IntervalPromotedType =
        typename IntervalPromotion<BareType<MathExpr>, BareType<EvalNumberType>>::Type;

    /// Check whether calling MathExpr with arguments ArgsPassed is an evaluation call (as opposed to a symbolic call).
    template<typename MathExpr, typename... ArgsPassed> struct IsEvaluationCall;

    /// Determine the resulting type for an evaluation call on a function MathExpr, given arguments ArgsPassed,
    /// using context Context; the Context allows overriding the evaluation number type and other options.
    template<typename Context, typename MathExpr, typename... ArgsPassed> struct EvaluationCallResult;
    template<typename Context, typename MathExpr, typename... ArgsPassed> using EvaluationCallResultType =
        typename EvaluationCallResult<Context,MathExpr,BareType<ArgsPassed>...>::Type;
    template<typename Context, typename ArrayType> struct ArrayEvaluationCallResult;
    template<typename Context, typename ArrayType> using ArrayEvaluationCallResultType =
        typename ArrayEvaluationCallResult<Context,ArrayType>::Type;

    /// A metafunction that finds the number type to use for evaluating CalledType given ArgsPassed as arguments.
    template<typename CalledType, typename... ArgsPassed> struct EvaluationNumberTypeImpl;
    template<typename CalledType, typename... ArgsPassed> using EvaluationNumberType =
        typename EvaluationNumberTypeImpl<CalledType, BareType<ArgsPassed>...>::Type;

    // The template that unpacks MathExpressions and does the actual evaluation.
    template<typename Context, typename CalledType, typename ArgArray> struct EvaluateImpl;

    /// Implementation template for evaluating predicates (or expressions that may occur as part of predicate eval).
    template<typename Context, typename CalledType, typename ArgArray, bool IsExpr = IsMathExpr<CalledType>::value>
        struct PredicateEvaluateImpl;

    template<typename Context> struct PredicateEvalResult;
    template<typename Context> using PredicateEvalResultType = typename PredicateEvalResult<Context>::Type;
}

    /// The default context to use when evaluating the given function or predicate with the given arguments.
    template<typename NumberType> struct DefaultContextWithNumberType;
    template<typename CalledType, typename... ArgsPassed> using DefaultEvaluationContext =
        DefaultContextWithNumberType<impl::EvaluationNumberType<CalledType,ArgsPassed...>>;
}
