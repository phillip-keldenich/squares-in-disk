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
// Created by Phillip Keldenich on 06.02.20.
//

#pragma once

#include "ivarp/metaprogramming.hpp"
#include "ivarp/tuple.hpp"
#include "ivarp/math_fn.hpp"
#include "ivarp/compile_time_bounds.hpp"
#include "ivarp/bound_dependency_analysis.hpp"
#include "ivarp/refactor_constraint_system/var_bounds.hpp"
#include "ivarp/refactor_constraint_system/maybe_bound.hpp"
#include "ivarp/refactor_constraint_system/labeled_bound.hpp"
#include "ivarp/refactor_constraint_system/bound_and_simplify.hpp"
#include "ivarp/refactor_constraint_system/constraints_to_bounds.hpp"
#include "ivarp/refactor_constraint_system/filter_constraints.hpp"
#include "ivarp/refactor_constraint_system/runtime_bound_table.hpp"
#include "ivarp/refactor_constraint_system/runtime_constraint_table.hpp"

namespace ivarp {
    /**
     * @struct ConstraintRefactoring
     * @brief Metafunction that runs one iteration of the constraint system refactoring process.
     *
     * It takes a tuple of variables and values with bounds and a tuple of constraints,
     * and produces:
     *  * A copy of the old constraint set with the given bounds applied, possibly extended and/or simplified.
     *  * A bound table: For each variable/value, a list of (compile-time) bound functions, labeled lb/ub/eq.
     *    This table only contains definitive bounds with known signs.
     *  * A list of bounded variables/values with new bound values.
     */
    struct ConstraintRefactoring {
        template<typename VarAndValBounds> struct ReboundConstraints {
        private:
            template<typename ConstrTuple, std::size_t I> using CAt =
                typename ConstrTuple::template At<I>;

            template<typename ConstrTuple, std::size_t I> using BSAt =
                impl::BoundAndSimplify<CAt<ConstrTuple,I>, VarAndValBounds>;

            template<typename ConstraintTuple, std::size_t... Indices>
                static inline auto rebound_constraints(ConstraintTuple&& ct, IndexPack<Indices...>) noexcept
            {
                return ivarp::make_tuple(
                    BSAt<ConstraintTuple,Indices>::apply(ivarp::template get<Indices>(ivarp::forward<ConstraintTuple>(ct)))...
                );
            }

        public:
            template<typename... Constraints>
            IVARP_H static inline auto rebound_constraints(Tuple<Constraints...>&& constraints) noexcept {
                 return rebound_constraints(ivarp::move(constraints), IndexRange<0,sizeof...(Constraints)>{});
            }

            /**
             * @brief The last simplify-and-bound step should use this metafunction instead;
             * constraints which are known to be satisfied will be filtered out in the process.
             */
             template<typename... Constraints>
             IVARP_H static inline auto rebound_and_filter_constraints(Tuple<Constraints...>&& constraints) noexcept {
                 auto unfiltered = rebound_constraints(ivarp::move(constraints));
                 return filter_constraints(ivarp::move(unfiltered));
             }
        };

        /// Create a table of bounds for each argument.
        template<typename VarAndValBounds, typename NewConstraints> struct NewBoundTable {
        private:
            template<std::size_t TargetIndex, std::size_t... CInds>
                static inline auto IVARP_H make_bound_table_for_arg(const NewConstraints& nc, IndexPack<CInds...>)
            {
                return concat_tuples(
                    (impl::rewrite_to_bound<TargetIndex, VarAndValBounds>(ivarp::template get<CInds>(nc)))...
                );
            }

            template<std::size_t TargetIndex, std::size_t... CInds> static inline auto IVARP_H
                make_bound_table_with_maybe_for_arg(const NewConstraints& nc, IndexPack<CInds...>)
            {
                return concat_tuples(
                    (impl::rewrite_to_bound_with_maybe<TargetIndex, VarAndValBounds>(ivarp::template get<CInds>(nc)))...
                );
            }

            template<std::size_t TargetIndex>
                static inline auto IVARP_H make_bound_table_for_arg(const NewConstraints& nc)
            {
                return make_bound_table_for_arg<TargetIndex>(nc, TupleIndexPack<NewConstraints>{});
            }

            template<std::size_t TargetIndex>
                static inline auto IVARP_H make_bound_table_with_maybe_for_arg(const NewConstraints& nc)
            {
                return make_bound_table_with_maybe_for_arg<TargetIndex>(nc, TupleIndexPack<NewConstraints>{});
            }

            template<std::size_t... AInds>
                static inline IVARP_H auto make_bound_table_impl(const NewConstraints& nc, IndexPack<AInds...>)
            {
                return ivarp::make_tuple(make_bound_table_for_arg<AInds>(nc)...);
            }

            template<std::size_t... AInds> static inline IVARP_H auto
                make_bound_table_with_maybe_impl(const NewConstraints& nc, IndexPack<AInds...>)
            {
                return ivarp::make_tuple(make_bound_table_with_maybe_for_arg<AInds>(nc)...);
            }

        public:
            static inline IVARP_H auto make_bound_table(const NewConstraints& nc) {
                return make_bound_table_impl(nc, TupleIndexPack<VarAndValBounds>{});
            }

            static inline IVARP_H auto make_bound_table_with_maybe(const NewConstraints& nc) {
                return make_bound_table_with_maybe_impl(nc, TupleIndexPack<VarAndValBounds>{});
            }

            using Type = decltype(make_bound_table(std::declval<NewConstraints>()));
        };

        /// Apply the new bounds to the arguments.
        template<typename VarAndValBounds, typename NewBoundTable>
            struct NewArgBounds
        {
        private:
            template<std::size_t TargetArg> struct NewBoundsForArg {
                using BoundTableForArg = typename NewBoundTable::template At<TargetArg>;
                using Bounds = typename impl::IntersectCTBTuple<TargetArg, VarAndValBounds, BoundTableForArg>::Bounds;
            };

            template<std::size_t... Inds>
                static inline Tuple<typename NewBoundsForArg<Inds>::Bounds...> new_bound_tuple_type(IndexPack<Inds...>);

        public:
            using ArgBounds = decltype(new_bound_tuple_type(TupleIndexPack<VarAndValBounds>{}));
        };
    };

    /**
     * @brief Refactor a constraint system.
     * @tparam InitialVarAndValBounds A tuple of compile time bounds,
     * @tparam Context The context.
     * @tparam Iterations The number of propapation iterations to perform at compile time.
     * @tparam Constraints
     * @param constraints A tuple of constraints.
     * @return A tuple [ RuntimeBoundTable, RuntimeConstraintTable,
     *                   Array<Context::NumberType> with runtime bounds, CompileTimeArgBounds ].
     */
    template<typename InitialVarAndValBounds, typename Context, std::size_t Iterations, typename Constraints>
        static inline auto refactor_constraints_with_iterations(Constraints&& constraints);

    /**
     * @brief Refactor a constraint system.
     *
     * Like refactor_constraints_with_iterations, performing 5 * NumArgs iterations.
     * @tparam InitialVarAndValBounds A tuple of compile time bounds,
     * @tparam Context The context.
     * @tparam Iterations The number of propapation iterations to perform at compile time.
     * @tparam Constraints
     * @param constraints A tuple of constraints.
     * @return A tuple [ RuntimeBoundTable, RuntimeConstraintTable,
     *                   Array<Context::NumberType> with runtime bounds, CompileTimeArgBounds ].
     */
    template<typename InitialVarAndValBounds, typename Context, typename Constraints>
        static inline auto refactor_constraints(Constraints&& constraints)
    {
        static constexpr std::size_t iterations = 5 * TupleSize<InitialVarAndValBounds>::value;
        return refactor_constraints_with_iterations<InitialVarAndValBounds,Context,iterations>(
            ivarp::forward<Constraints>(constraints)
        );
    }
}

#include "ivarp/refactor_constraint_system/refactor_constraints_impl.hpp"
