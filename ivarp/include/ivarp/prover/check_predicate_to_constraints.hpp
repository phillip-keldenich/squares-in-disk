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
// Created by Phillip Keldenich on 16.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename CheckPredicate> struct CheckPredicateToConstraintsImpl {
        using Type = Tuple<UnaryMathPred<UnaryMathPredNotTag, CheckPredicate>>;

        static Type predicate_constraints(const CheckPredicate& p) {
            return Type{!p};
        }
    };

    template<typename Tag, typename... Args> struct CheckPredicateToConstraintsOrImpl {
        using CheckPredicate = NAryMathPred<Tag, Args...>;
        using Type = Tuple<UnaryMathPred<UnaryMathPredNotTag, Args>...>;

        static Type predicate_constraints(const CheckPredicate& p) {
            return compute_predicate_constraints(p, IndexRange<0,sizeof...(Args)>{});
        }

    private:
        template<std::size_t... Inds>
            static Type compute_predicate_constraints(const CheckPredicate& p, IndexPack<Inds...>)
        {
            return {(!get<Inds>(p.args))...};
        }
    };

    template<typename... Args> struct CheckPredicateToConstraintsImpl<NAryMathPred<MathPredOr, Args...>> :
        CheckPredicateToConstraintsOrImpl<MathPredOr, Args...>
    {};

    template<typename... Args> struct CheckPredicateToConstraintsImpl<NAryMathPred<MathPredOrSeq, Args...>> :
        CheckPredicateToConstraintsOrImpl<MathPredOrSeq, Args...>
    {};

    template<typename CheckPredicate> using CheckPredicateToConstraints =
        typename CheckPredicateToConstraintsImpl<std::decay_t<CheckPredicate>>::Type;

    template<typename CheckPredicate> static inline CheckPredicateToConstraints<CheckPredicate>
        check_predicate_to_constraints(const CheckPredicate& c)
    {
        return CheckPredicateToConstraintsImpl<std::decay_t<CheckPredicate>>::predicate_constraints(c);
    }

    template<typename CheckPredConstraints, typename... ConstraintPreds> struct ConstraintsAndPredicateImpl;
    template<typename... CheckPredConstraints, typename... ConstraintPreds>
        struct ConstraintsAndPredicateImpl<Tuple<CheckPredConstraints...>, ConstraintPreds...>
    {
        using ConstraintsType = Constraints<CheckPredConstraints..., ConstraintPreds...>;

        template<typename CheckPred, std::size_t... CPInds>
            static ConstraintsType constraints(const CheckPred& cp, ConstraintPreds... constrs, IndexPack<CPInds...>)
        {
            auto cpconstrs = impl::check_predicate_to_constraints(cp);
            return ConstraintsType{ std::move(get<CPInds>(cpconstrs))..., std::move(constrs)... };
        }
    };
}

    template<typename CheckPred, typename... ConstraintPreds> static inline
        typename impl::ConstraintsAndPredicateImpl<impl::CheckPredicateToConstraints<CheckPred>,
                                                   ConstraintPreds...>::ConstraintsType
            predicate_and_constraints_no_folding(const CheckPred& pred, const ConstraintPreds&... constrs)
    {
        using CheckConstrTuple = impl::CheckPredicateToConstraints<CheckPred>;
        return impl::ConstraintsAndPredicateImpl<CheckConstrTuple, ConstraintPreds...>::
            constraints(pred, constrs..., IndexRange<0,TupleSize<CheckConstrTuple>::value>{});
    }

    template<typename CheckPred, typename... ConstraintPreds> static inline auto
        predicate_and_constraints(const CheckPred& pred, const ConstraintPreds&... constrs)
    {
        return predicate_and_constraints_no_folding(fold_constants(pred), fold_constants(constrs)...);
    }
}
