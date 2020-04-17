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
// Created by Phillip Keldenich on 06.11.19.
//

#pragma once

#include "ivarp/constraint_reformulation.hpp"

namespace ivarp {
namespace impl {
    /// The empty default implementation of a constraint rewritten as bound.
    template<typename ConstraintType, std::size_t ConstraintIndex, std::size_t VariableIndex,
             bool CanReformulate=CanReformulateIntoBound<ConstraintType, MathArgFromIndex<VariableIndex>>::value>
        struct ConstraintToBoundImpl
    {
        explicit ConstraintToBoundImpl(const ConstraintType& /*c*/) noexcept {}
    };

    /// Implementation of a constraint rewritten as bound, for constraints which can
    /// (at least potentially) be rewritten as bounds.
    template<typename ConstraintType_, std::size_t ConstraintIndex, std::size_t VariableIndex>
        struct ConstraintToBoundImpl<ConstraintType_, ConstraintIndex, VariableIndex, true>
    {
        using ConstraintType = ConstraintType_;
        static constexpr std::size_t constraint_index = ConstraintIndex;
        static constexpr std::size_t variable_index = VariableIndex;
        using VariableArg = MathArgFromIndex<VariableIndex>;
        using Reformulator = ReformulateIntoBound<ConstraintType, VariableArg>;
        using BoundFunction = typename Reformulator::BoundFunctionType;

        explicit ConstraintToBoundImpl(const ConstraintType& c) :
            bound_function(Reformulator::bound_function(c))
        {}

        /// Check (at runtime, potentially using the actual variable values) whether rewriting was successful.
        template<typename Context, typename ConstraintsType, typename ArgArray>
            BoundDirection bound_direction(const ConstraintsType& c, const ArgArray& args) const
        {
            const auto& constraint = get<constraint_index>(c.constraints);
            return Reformulator::template bound_direction<Context>(constraint, bound_function, args);
        }

        template<typename Context, typename ConstraintsType, typename ArgArray>
            ApplyBoundResult apply_bound(BoundDirection direction, const ConstraintsType& /*c*/, ArgArray& args) const
        {
            if(direction == BoundDirection::NONE) {
                return ApplyBoundResult{false, false};
            }

            auto bound_result = bound_function.template array_evaluate<Context>(const_cast<const ArgArray&>(args));
            if(bound_result.possibly_undefined()) {
                return ApplyBoundResult{false, false};
            }

            ApplyBoundResult result{false,false};
            if((direction & BoundDirection::LEQ) != BoundDirection::NONE) {
                result |= get<variable_index>(args).restrict_upper_bound(bound_result);
            }
            if((direction & BoundDirection::GEQ) != BoundDirection::NONE) {
                result |= get<variable_index>(args).restrict_lower_bound(bound_result);
            }
            return result;
        }

        /// Check whether the bound was successfully rewritten and apply it to the given variable values.
        /// Returns whether the corresponding variable bounds were changed (or even became empty).
        template<typename Context, typename ConstraintsType, typename ArgArray>
            ApplyBoundResult apply_bound(const Context&, const ConstraintsType& c, ArgArray& args) const
        {
            BoundDirection direction = bound_direction<Context>(c, args);
            return apply_bound<Context>(direction, c, args);
        }

        BoundFunction bound_function;
    };

    /// List variable indices which we can bound-rewrite certain constraints to.
    template<typename ConstraintType_, std::size_t NumVars, std::size_t NotVar, std::size_t CurVar, typename CurPack>
        struct RewritableVarsImpl;

    template<typename ConstraintType, typename CurPack, std::size_t NotVar, std::size_t CurVar, std::size_t NumVars,
             bool CanRewrite = CurVar != NotVar && CanReformulateIntoBound<ConstraintType, MathArgFromIndex<CurVar>>::value>
        struct RewritableVarsStepImpl
    {
        using Type = typename RewritableVarsImpl<ConstraintType, NumVars, NotVar, CurVar+1, CurPack>::Type;
    };

    template<typename ConstraintType, typename CurPack, std::size_t NotVar, std::size_t CurVar, std::size_t NumVars>
        struct RewritableVarsStepImpl<ConstraintType, CurPack, NotVar, CurVar, NumVars, true>
    {
        using Type = typename RewritableVarsImpl<ConstraintType, NumVars, NotVar, CurVar+1,
                                                 typename CurPack::template Append<CurVar>::Type>::Type;
    };

    template<typename ConstraintType_, std::size_t NumVars, std::size_t NotVar, std::size_t CurVar, typename CurPack>
        struct RewritableVarsImpl
    {
        static_assert(IsMathPred<ConstraintType_>::value, "RewritableVars given a type that is not a constraint!");
        using Type = typename RewritableVarsStepImpl<ConstraintType_, CurPack, NotVar, CurVar, NumVars>::Type;
    };

    template<typename ConstraintType_, std::size_t NumVars, std::size_t NotVar, typename CurPack>
        struct RewritableVarsImpl<ConstraintType_, NumVars, NotVar, NumVars, CurPack>
    {
        static_assert(IsMathPred<ConstraintType_>::value, "RewritableVars given a type that is not a constraint!");
        using Type = CurPack;
    };
}

    template<typename... ConstraintPreds> struct Constraints {
        static_assert(AllOf<IsMathPred<ConstraintPreds>::value...>::value, "constraints was passed a non-predicate!");

        static constexpr std::size_t num_constraints = sizeof...(ConstraintPreds);

        explicit Constraints(ConstraintPreds&&... cstrs) :
            constraints(cstrs...),
            constraints_to_bounds(make_constraints_to_bounds(constraints))
        {}

        using ConstraintTuple = Tuple<ConstraintPreds...>;

        /// Get the function type of the constraint with the given Index.
        template<std::size_t Index> using ByIndex = TupleElementType<Index, ConstraintTuple>;

        /// Get an IndexPack with all constraint indices that depend on the variable with the given VarIndex.
        template<std::size_t VarIndex> struct DependingOnVarImpl {
            template<typename T> using Filter = impl::DependsOnArgIndex<T, VarIndex>;
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t VarIndex> using DependingOnVar = typename DependingOnVarImpl<VarIndex>::Type;

        /// Get an IndexPack with all constraint indices that can be
        /// applied when only the first NumVars variables are given.
        template<std::size_t NumVars> struct ApplicableWithVarsImpl {
            template<typename T> struct Filter {
                static constexpr bool value = NumArgs<T>::value <= NumVars;
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t NumVars> using ApplicableWithVars = typename ApplicableWithVarsImpl<NumVars>::Type;

        /// Get an IndexPack with all constraint indices that can be _newly_
        /// applied once the first NumVars variables are given.
        template<std::size_t NumVars> struct NewlyApplicableWithVarsImpl {
            template<typename T> struct Filter {
                static constexpr bool value = NumArgs<T>::value == NumVars;
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t NumVars> using NewlyApplicableWithVars =
            typename NewlyApplicableWithVarsImpl<NumVars>::Type;

        /// Get an IndexPack with all constraints that may be used to compute bounds for a new variable.
        template<std::size_t NewVar> struct NewVariableBoundConstraintsImpl {
            template<typename T> struct Filter {
                static constexpr bool value = NumArgs<T>::value == NewVar+1 &&
                                              CanReformulateIntoBound<T, MathArgFromIndex<NewVar>>::value;
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t NewVar> using NewVariableBoundConstraints =
            typename NewVariableBoundConstraintsImpl<NewVar>::Type;

        /// Get an IndexPack with all constraints that cannot be rewritten to bounds for a new variable,
        /// but may still be _newly_ used to prune once that variable is included.
        template<std::size_t NewVar> struct NewVariablePruneConstraintsImpl {
            template<typename T> struct Filter {
                static constexpr bool value = NumArgs<T>::value == NewVar+1 &&
                                              !CanReformulateIntoBound<T, MathArgFromIndex<NewVar>>::value;
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t NewVar> using NewVariablePruneConstraints =
            typename NewVariablePruneConstraintsImpl<NewVar>::Type;

        /// Get an IndexPack with all variable indices (except NotVar) from [0,NumVars) that the
        /// given constraint can possibly be bound-rewritten to.
        template<typename Constraint, std::size_t NumVars, std::size_t NotVar=~std::size_t(0)> using RewritableVariables
            = typename impl::RewritableVarsImpl<Constraint,NumVars,NotVar,0,IndexPack<>>::Type;


        /// Get an IndexPack with all constraints that satisfy all of the following:
        ///  * Can be applied when NumVars are present,
        ///  * Can be bound-rewritten for at least one variable that is not ChangedVariable,
        ///  * Depend on the given ChangedVariable.
        template<std::size_t NumVars, std::size_t ChangedVariable> struct BoundPropagationConstraintsImpl {
            template<typename T> struct Filter {
                using RewritableVars = RewritableVariables<T, NumVars, ChangedVariable>;

                static constexpr bool value = NumArgs<T>::value <= NumVars &&
                                              impl::DependsOnArgIndex<T,ChangedVariable>::value &&
                                              (RewritableVars::size > 0);
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t NumVars, std::size_t ChangedVariable> using BoundPropagationConstraints =
            typename BoundPropagationConstraintsImpl<NumVars, ChangedVariable>::Type;

        /// Get an IndexPack with all constraints that satisfy all of the following:
        ///   * Can be applied when NumVars are present,
        ///   * Can be bound-rewritten to TargetVar.
        template<std::size_t TargetVar, std::size_t NumVars> struct RestrictBoundConstraintsImpl {
            template<typename T> struct Filter {
                static constexpr bool value = NumArgs<T>::value <= NumVars &&
                                              CanReformulateIntoBound<T,MathArgFromIndex<TargetVar>>::value;
            };
            using Type = FilteredIndexPackType<Filter, ConstraintPreds...>;
        };
        template<std::size_t TargetVar, std::size_t NumVars> using RestrictBoundConstraints =
            typename RestrictBoundConstraintsImpl<TargetVar,NumVars>::Type;

        /// The constraint-to-bound transformation type for the given constraint (ConstraintType,ConstraintIndex)
        /// and the target variable VariableIndex.
        template<typename ConstraintType, std::size_t ConstraintIndex, std::size_t VariableIndex>
            using ConstraintToBound = impl::ConstraintToBoundImpl<ConstraintType, ConstraintIndex, VariableIndex>;

        /// The tuple of constraint-to-bound transformation types for the given ConstraintIndex.
        template<std::size_t ConstraintIndex> struct ConstraintToBoundInnerTupleImpl {
            using ConstraintType = ByIndex<ConstraintIndex>;
            template<std::size_t VariableIndex> using CTB =
                ConstraintToBound<ConstraintType, ConstraintIndex, VariableIndex>;
            using Type = ForEachIndexType<Tuple, CTB, IndexRange<0, NumArgs<ConstraintType>::value>>;

            static Type make_tuple(const ConstraintTuple& c) {
                return make_tuple(c, IndexRange<0,NumArgs<ConstraintType>::value>{});
            }

            template<std::size_t... Indices> static Type make_tuple(const ConstraintTuple& c, IndexPack<Indices...>) {
                return Type{(CTB<Indices>(get<ConstraintIndex>(c)))...};
            }
        };
        template<std::size_t ConstraintIndex> using ConstraintToBoundInnerTuple =
            typename ConstraintToBoundInnerTupleImpl<ConstraintIndex>::Type;

        /// A tuple containing the constraint-to-bound types for all constraint/variable pairs.
        using ConstraintToBoundOuterTuple =
            ForEachIndexType<Tuple, ConstraintToBoundInnerTuple, IndexRange<0, num_constraints>>;

        /// For each constraint, store all possible reformulations as variable bounds in a tuple of tuples.
        /// This method creates that tuple.
        static ConstraintToBoundOuterTuple make_constraints_to_bounds(const ConstraintTuple& constraints) {
            return make_constraints_to_bounds(constraints, IndexRange<0,num_constraints>{});
        }
        template<std::size_t... Indices> static ConstraintToBoundOuterTuple
            make_constraints_to_bounds(const ConstraintTuple& constraints, IndexPack<Indices...>)
        {
            return ConstraintToBoundOuterTuple{ConstraintToBoundInnerTupleImpl<Indices>::make_tuple(constraints)...};
        }

        /// Store all constraints in a tuple type.
        ConstraintTuple constraints;
        /// For each constraint, store all possible reformulations as variable bounds in a tuple of tuples.
        ConstraintToBoundOuterTuple constraints_to_bounds;
    };

    template<typename T> struct IsConstraints : std::false_type {};
    template<typename... Preds> struct IsConstraints<Constraints<Preds...>> : std::true_type {};

    /// Create a constraints object from a sequence of predicates.
    template<typename... ConstraintPreds> static inline Constraints<std::decay_t<ConstraintPreds>...>
        constraints(ConstraintPreds&&... cstrs)
    {
        return Constraints<std::decay_t<ConstraintPreds>...>{std::forward<ConstraintPreds>(cstrs)...};
    }
}
