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
// Created by Phillip Keldenich on 04.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename VarAndValBounds, typename Constraints, std::size_t Iterations,
             std::enable_if_t<Iterations == 0, int> = 0>
        static inline auto refactor_constraints_impl(std::remove_reference_t<Constraints>&& constraints)
    {
        auto new_constraints = ConstraintRefactoring::ReboundConstraints<VarAndValBounds>::
                                                      rebound_and_filter_constraints(ivarp::move(constraints));
        using NewConstraints = decltype(new_constraints);
        auto new_bt = ConstraintRefactoring::NewBoundTable<VarAndValBounds, NewConstraints>::
                                             make_bound_table_with_maybe(new_constraints);
        auto rbt = make_runtime_bound_table(ivarp::move(new_bt));
        auto rct = make_runtime_constraint_table<VarAndValBounds>(ivarp::move(new_constraints));
        return ivarp::make_tuple(ivarp::move(rbt), ivarp::move(rct), VarAndValBounds{});
    }

    template<typename VarAndValBounds, typename Constraints, std::size_t Iterations,
             std::enable_if_t<(Iterations > 0),int> = 0>
        static inline auto refactor_constraints_impl(std::remove_reference_t<Constraints>&& constraints)
    {
        auto new_constraints = ConstraintRefactoring::ReboundConstraints<VarAndValBounds>::rebound_constraints(
            ivarp::move(constraints)
        );

        using NewConstraints = decltype(new_constraints);
        using NewBTType = typename ConstraintRefactoring::NewBoundTable<VarAndValBounds, NewConstraints>::Type;
        using NewArgBounds = typename ConstraintRefactoring::NewArgBounds<VarAndValBounds, NewBTType>::ArgBounds;

        return refactor_constraints_impl<NewArgBounds, NewConstraints, Iterations-1>(ivarp::move(new_constraints));
    }

    template<typename InitialVarAndValBounds, std::size_t Iterations, typename Constraints>
    static inline auto do_refactor_constraints(std::remove_reference_t<Constraints>&& constraints) {
        return refactor_constraints_impl<InitialVarAndValBounds, Constraints, Iterations>(ivarp::move(constraints));
    }

    template<typename InitialVarAndValBounds, std::size_t Iterations, typename Constraints>
    static inline auto do_refactor_constraints(const std::remove_reference_t<Constraints>& constraints) {
        BareType<Constraints> copy(constraints);
        return refactor_constraints_impl<InitialVarAndValBounds, BareType<Constraints>, Iterations>(ivarp::move(copy));
    }

    template<typename IntervalType, typename CTBounds, std::size_t... A>
    static inline IVARP_H auto initial_runtime_bounds(IndexPack<A...>) {
        using fixed_point_bounds::fp_to_rational_interval;
        return make_array<IntervalType>(
            mark_defined(convert_number<IntervalType>(fp_to_rational_interval(
                CTBounds::template At<A>::lb,
                CTBounds::template At<A>::ub
            )))...
        );
    }

    template<typename Context, typename RuntimeBoundTable, typename Bounds> static inline IVARP_H void
        iterate_bounding(std::size_t iterations, Bounds& bounds, const RuntimeBoundTable& btable)
    {
        for(std::size_t i = 0; i < iterations; ++i) {
            btable.template apply_all_bounds<Context>(bounds);
        }
    }

    template<typename Context, typename RuntimeBoundTable, typename CTBounds>
    static inline IVARP_H auto make_runtime_bounds(const RuntimeBoundTable& bound_table, const CTBounds&)
    {
        using ArgIndexPack = TupleIndexPack<CTBounds>;
        using IntervalType = typename Context::NumberType;
        constexpr std::size_t numargs = TupleSize<CTBounds>::value;

        static_assert(IsIntervalType<IntervalType>::value, "make_runtime_bounds requires an interval type!");
        Array<IntervalType, numargs> result = initial_runtime_bounds<IntervalType, CTBounds>(ArgIndexPack{});
        iterate_bounding<Context>(50 * numargs, result, bound_table);
        return result;
    }
}
}

template<typename InitialVarAndValBounds, typename Context, std::size_t Iterations, typename Constraints>
static inline auto ivarp::refactor_constraints_with_iterations(Constraints&& constraints)
{
    // [ BoundTable, ConstraintTable, CompileTimeBounds ]
    auto tables = impl::template do_refactor_constraints<InitialVarAndValBounds, Iterations, BareType<Constraints>>(
            ivarp::forward<Constraints>(constraints)
    );

    // array of Context::NumberType
    auto runtime_bounds = impl::make_runtime_bounds<Context>(
            ivarp::template get<0>(tables), ivarp::template get<2>(tables));

    return ivarp::make_tuple(
        ivarp::template get<0>(ivarp::move(tables)), ivarp::template get<1>(ivarp::move(tables)),
        ivarp::move(runtime_bounds), ivarp::template get<2>(tables)
    );
}
