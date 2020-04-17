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

/** 
 * @file n_ary_ops.hpp
 * Definition of logical junctors between predicates/bools.
 * We ensure that any combination like (a | (b | c)), a | b | c | d | e is turned into a single NAry operator.
 */

#pragma once

namespace ivarp {
namespace impl {
    template<typename... Bounds> struct AndEvalBoundsImpl {
        static constexpr bool lb = AllOf<Bounds::lb...>::value;
        static constexpr bool ub = AllOf<Bounds::ub...>::value;
    };

    template<typename... Bounds> struct OrEvalBoundsImpl {
        static constexpr bool lb = OneOf<Bounds::lb...>::value;
        static constexpr bool ub = OneOf<Bounds::ub...>::value;
    };
}

    /// Tag for |.
    struct MathPredOr : MathPredJunctor {
        static constexpr PrintOp print_operator = PrintOp::OR;

        struct EvalBounds {
            template<typename... Bounds> using Eval = impl::OrEvalBoundsImpl<Bounds...>;
        };

        template<typename Context, typename Arg1>
            static inline impl::PredicateEvalResultType<Context> eval(const Arg1& arg1) noexcept
        {
            return arg1;
        }

        template<typename Context, typename Arg1, typename Arg2, typename... Args> static inline
            impl::PredicateEvalResultType<Context>
                eval(const Arg1& arg1, const Arg2& arg2, const Args&... args) noexcept
        {
            return arg1 | eval<Context>(arg2, args...);
        }
    };

    /// Tag for ||. Implementation is done in predicate_eval.hpp due to sequencing semantics.
    struct MathPredOrSeq : MathPredJunctor, MathPredSeq {
        static constexpr PrintOp print_operator = PrintOp::OR;

        struct EvalBounds {
            template<typename... Bounds> using Eval = impl::OrEvalBoundsImpl<Bounds...>;
        };
    };

    /// Tag for &.
    struct MathPredAnd : MathPredJunctor {
        static constexpr PrintOp print_operator = PrintOp::AND;

        struct EvalBounds {
            template<typename... Bounds> using Eval = impl::AndEvalBoundsImpl<Bounds...>;
        };

        template<typename Context, typename Arg1>
            static inline impl::PredicateEvalResultType<Context> eval(const Arg1& arg1) noexcept
        {
            return arg1;
        }

        template<typename Context, typename Arg1, typename Arg2, typename... Args> static inline
            impl::PredicateEvalResultType<Context>
                eval(const Arg1& arg1, const Arg2& arg2, const Args&... args) noexcept
        {
            return arg1 & eval<Context>(arg2, args...);
        }
    };

    /// Tag for &&. Implementation is done in predicate_eval.hpp due to sequencing semantics.
    struct MathPredAndSeq : MathPredJunctor, MathPredSeq {
        static constexpr PrintOp print_operator = PrintOp::AND;

        struct EvalBounds {
            template<typename... Bounds> using Eval = impl::AndEvalBoundsImpl<Bounds...>;
        };
    };

    /// Check whether the given type is an N-ary predicate with the given tag.
    template<typename T, typename Tag> struct IsNAryPredWithTagImpl : std::false_type {};
    template<typename Tag, typename... Args>
        struct IsNAryPredWithTagImpl<NAryMathPred<Tag, Args...>, Tag> : std::true_type {};
    template<typename T, typename Tag> using IsNAryPredWithTag = IsNAryPredWithTagImpl<std::decay_t<T>, Tag>;

namespace impl {
    template<typename LHS, typename RHS, typename OpTag,
             bool ArgsOk = (IsMathPred<LHS>::value || IsBoolean<LHS>::value) && // both are either boolean or predicates
                           (IsMathPred<RHS>::value || IsBoolean<RHS>::value) &&
                           (IsMathPred<LHS>::value || IsMathPred<RHS>::value)>  // and at least one is a predicate
    struct LogicJunctor {};

    /// Handling the case where neither LHS nor RHS are NAry predicates with the right tag.
    template<typename LHS, typename RHS, typename OpTag>
        struct LogicJunctorMergeImpl
    {
        using Type = NAryMathPred<OpTag, EnsurePred<LHS>, EnsurePred<RHS>>;

        template<typename L, typename R>
            static std::enable_if_t<!IsBoolean<L>::value && !IsBoolean<R>::value, Type> merge(L&& l, R&& r)
        {
            return {std::forward<L>(l), std::forward<R>(r)};
        }

		template<typename L, typename R>
            static std::enable_if_t<IsBoolean<L>::value && !IsBoolean<R>::value, Type> merge(L&& l, R&& r)
        {
            return {constant(l), std::forward<R>(r)};
        }

		template<typename L, typename R>
            static std::enable_if_t<!IsBoolean<L>::value && IsBoolean<R>::value, Type> merge(L&& l, R&& r)
        {
            return {std::forward<L>(l), constant(r)};
        }

		template<typename L, typename R>
            static std::enable_if_t<IsBoolean<L>::value && IsBoolean<R>::value, Type> merge(L&& l, R&& r)
        {
            return {constant(l), constant(r)};
        }
    };

    /// Handling the case where LHS is an NAry predicate with the right tag, but RHS is not.
    template<typename... LHSArgs, typename RHS, typename OpTag>
        struct LogicJunctorMergeImpl<NAryMathPred<OpTag, LHSArgs...>, RHS, OpTag>
    {
        using Type = NAryMathPred<OpTag, LHSArgs..., EnsurePred<RHS>>;

    private:
        using LHSType = NAryMathPred<OpTag, LHSArgs...>;

        template<typename R, std::size_t... OldArgIndices>
            static Type do_merge(const LHSType& l, R&& r, IndexPack<OldArgIndices...>)
        {
            return {get<OldArgIndices>(l.args)..., std::forward<R>(r)};
        }

    public:
        template<typename R>
            static Type merge(const LHSType& l, R&& r)
        {
            return do_merge(l, std::forward<R>(r), TupleIndexPack<typename LHSType::Args>{});
        }
    };

    /// Symmetrically, handling the case where RHS is an NAryMathPred with the right tag but LHS is not.
    template<typename... RHSArgs, typename LHS, typename OpTag>
        struct LogicJunctorMergeImpl<LHS, NAryMathPred<OpTag, RHSArgs...>, OpTag>
    {
        using Type = NAryMathPred<OpTag, EnsurePred<LHS>, RHSArgs...>;

    private:
        using RHSType = NAryMathPred<OpTag, RHSArgs...>;

        template<typename L, std::size_t... OldArgIndices>
            static Type do_merge(L&& l, const RHSType& r, IndexPack<OldArgIndices...>)
        {
            return {std::forward<L>(l), get<OldArgIndices>(r.args)...};
        }

    public:
        template<typename L>
            static Type merge(L&& l, const RHSType& r)
        {
            return do_merge(std::forward<L>(l), r, TupleIndexPack<typename RHSType::Args>{});
        }
    };

    /// Finally, handling the case where both LHS and RHS are NAryMathPred with the right tag.
    template<typename... LHSArgs, typename... RHSArgs, typename OpTag>
        struct LogicJunctorMergeImpl<NAryMathPred<OpTag, LHSArgs...>, NAryMathPred<OpTag, RHSArgs...>, OpTag>
    {
        using Type = NAryMathPred<OpTag, LHSArgs..., RHSArgs...>;

    private:
        using LHSType = NAryMathPred<OpTag, LHSArgs...>;
        using RHSType = NAryMathPred<OpTag, RHSArgs...>;

        template<std::size_t... LHSArgIndices, std::size_t... RHSArgIndices>
            static Type do_merge(const LHSType& l, const RHSType& r,
                                 IndexPack<LHSArgIndices...>, IndexPack<RHSArgIndices...>)
        {
            return {get<LHSArgIndices>(l.args)..., get<RHSArgIndices>(r.args)...};
        }

    public:
        static Type merge(const LHSType& l, const RHSType& r) {
            return do_merge(l, r, TupleIndexPack<typename LHSType::Args>{}, TupleIndexPack<typename RHSType::Args>{});
        }
    };

    template<typename LHS, typename RHS, typename OpTag> struct LogicJunctor<LHS,RHS,OpTag,true> :
        LogicJunctorMergeImpl<std::decay_t<LHS>,std::decay_t<RHS>,OpTag>
    {};
}

    /// Other logic operators (|,||,&,&&), all N-ary.
#define IVARP_DEFINE_NARY_LOGIC_OP(op, Tag)\
    template<typename LHS, typename RHS> static inline\
        typename impl::LogicJunctor<LHS,RHS,Tag>::Type\
            operator op(LHS&& lhs, RHS&& rhs)\
    {\
        return impl::LogicJunctor<LHS,RHS,Tag>::merge(std::forward<LHS>(lhs), std::forward<RHS>(rhs));\
    }

    IVARP_DEFINE_NARY_LOGIC_OP(|, MathPredOr)
    IVARP_DEFINE_NARY_LOGIC_OP(||, MathPredOrSeq)
    IVARP_DEFINE_NARY_LOGIC_OP(&, MathPredAnd)
    IVARP_DEFINE_NARY_LOGIC_OP(&&, MathPredAndSeq)
#undef IVARP_DEFINE_NARY_LOGIC_OP
}
