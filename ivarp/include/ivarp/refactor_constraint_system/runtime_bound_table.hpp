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
// Created by Phillip Keldenich on 30.01.20.
//

#pragma once

#include "ivarp/bound_propagation/propagation_result.hpp"
#include "ivarp/bound_propagation/bound_event.hpp"

namespace ivarp {
namespace impl {
    template<std::size_t ArgIndex, typename ArgBounds, std::size_t... BInds>
        static inline auto make_rbt_tag_arg_bounds(ArgBounds&& bounds, IndexPack<BInds...>)
    {
        return ivarp::make_tuple(label_bound<ArgIndex>(ivarp::template get<BInds>(ivarp::forward<ArgBounds>(bounds)))...);
    }

    template<std::size_t ArgIndex, typename ArgBounds>
        static inline auto make_rbt_tag_arg_bounds(ArgBounds&& bounds)
    {
        return make_rbt_tag_arg_bounds<ArgIndex>(ivarp::forward<ArgBounds>(bounds), TupleIndexPack<ArgBounds>{});
    }

    template<typename BoundsByArg, std::size_t... ArgInds>
        static inline auto make_rbt_bounds(BoundsByArg&& argbounds, IndexPack<ArgInds...>)
    {
        return concat_tuples(
            (make_rbt_tag_arg_bounds<ArgInds>(ivarp::template get<ArgInds>(ivarp::forward<BoundsByArg>(argbounds))))...
        );
    }

    template<typename BoundType, std::size_t ChangedArgIndex> struct NeedsUpdate;
    template<typename BoundType, std::size_t Label, std::size_t ChangedArgIndex>
        struct NeedsUpdate<LabeledBound<BoundType, Label>, ChangedArgIndex> :
            NeedsUpdate<BoundType, ChangedArgIndex>
    {};
    template<typename BoundFunction, BoundDirection Direction, std::size_t ChangedArgIndex>
        struct NeedsUpdate<CompileTimeBound<BoundFunction, Direction>, ChangedArgIndex>
    {
    private:
        static constexpr BoundDependencies deps = compute_bound_dependencies<BoundFunction,ChangedArgIndex>();
        static constexpr bool needs_bound_lb = (Direction == BoundDirection::GEQ) ||
                                               (Direction == BoundDirection::BOTH);
        static constexpr bool needs_bound_ub = (Direction == BoundDirection::LEQ) ||
                                               (Direction == BoundDirection::BOTH);

    public:
        static constexpr bool on_lb_changed = (needs_bound_lb && deps.lb_depends_on_lb) ||
                                              (needs_bound_ub && deps.ub_depends_on_lb);
        static constexpr bool on_ub_changed = (needs_bound_lb && deps.lb_depends_on_ub) ||
                                              (needs_bound_ub && deps.ub_depends_on_ub);
    };
    template<typename BoundFunction, typename CheckDirection, std::size_t ChangedArgIndex>
        struct NeedsUpdate<MaybeBound<BoundFunction,CheckDirection>, ChangedArgIndex>
    {
    private:
        static constexpr BoundDependencies deps = compute_bound_dependencies<BoundFunction,ChangedArgIndex>();
        static constexpr BoundID check_deps = CheckDirection::template compute_bound_dependencies<ChangedArgIndex>();

    public:
        static constexpr bool on_lb_changed = (deps.lb_depends_on_lb || deps.ub_depends_on_lb ||
                                               check_deps == BoundID::LB || check_deps == BoundID::BOTH);
        static constexpr bool on_ub_changed = (deps.lb_depends_on_ub || deps.ub_depends_on_ub ||
                                               check_deps == BoundID::UB || check_deps == BoundID::BOTH);
    };

    template<std::size_t ArgCount, typename BoundTable> struct OnUpdateInfoImpl {
    private:
        using AllBoundInds = TupleIndexPack<BoundTable>;
        using AllArgInds = IndexRange<0, ArgCount>;

        template<std::size_t ChangedArg> struct Filter {
            template<std::size_t Bound>
                struct NeedsUpdateOnLBChange
            {
                using BoundType = typename BoundTable::template At<Bound>;
                static constexpr bool value = NeedsUpdate<BoundType, ChangedArg>::on_lb_changed;
            };
            template<std::size_t Bound>
                struct NeedsUpdateOnUBChange
            {
                using BoundType = typename BoundTable::template At<Bound>;
                static constexpr bool value = NeedsUpdate<BoundType, ChangedArg>::on_ub_changed;
            };
        };

        template<std::size_t ChangedArg> using UpdateBoundsOnLBChange =
            FilterIndexPack<Filter<ChangedArg>::template NeedsUpdateOnLBChange, AllBoundInds>;
        template<std::size_t ChangedArg> using UpdateBoundsOnUBChange =
            FilterIndexPack<Filter<ChangedArg>::template NeedsUpdateOnUBChange, AllBoundInds>;
        template<std::size_t ChangedArg> using UpdateBoundsOnBothChange =
            MergeIndexPacks<UpdateBoundsOnLBChange<ChangedArg>, UpdateBoundsOnUBChange<ChangedArg>>;

        template<std::size_t... ArgInds>
            static Tuple<UpdateBoundsOnLBChange<ArgInds>...> on_update_lb_type(IndexPack<ArgInds...>);
        template<std::size_t... ArgInds>
            static Tuple<UpdateBoundsOnUBChange<ArgInds>...> on_update_ub_type(IndexPack<ArgInds...>);
        template<std::size_t... ArgInds>
            static Tuple<UpdateBoundsOnBothChange<ArgInds>...> on_update_both_type(IndexPack<ArgInds...>);

    public:
        struct TypeContainer {
            using OnLBChange = decltype(on_update_lb_type(AllArgInds{}));
            using OnUBChange = decltype(on_update_ub_type(AllArgInds{}));
            using OnBothChange = decltype(on_update_both_type(AllArgInds{}));
        };
    };
}

    template<std::size_t ArgCount, typename BoundTable> struct RuntimeBoundTable {
        IVARP_DEFAULT_CM(RuntimeBoundTable);

        RuntimeBoundTable(BoundTable&& bt) noexcept :
            table(ivarp::move(bt))
        {}

        template<std::size_t BoundIndex> using At = typename BoundTable::template At<BoundIndex>;

        static constexpr std::size_t num_bounds = TupleSize<BoundTable>::value;
        BoundTable table;
        struct OnUpdateInfo : impl::OnUpdateInfoImpl<ArgCount, BoundTable>::TypeContainer {};

        // Apply all bounds; this is used during initial constraint propagation.
        template<typename Context, typename BoundValueArray>
            PropagationResult apply_all_bounds(BoundValueArray& bounds) const
        {
            return do_apply_all_bounds<Context>(bounds, TupleIndexPack<BoundTable>{});
        }

        /// Apply a single bound. The returned BoundID is a LabeledBoundID with the target arg.
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(std::size_t BoundIndex, typename Context, typename BoundValueArray),
            typename Context::NumberType,
            auto apply_bound(BoundValueArray& bounds) const noexcept(IsCUDANumber<typename Context::NumberType>::value)
            {
                const auto& id = ivarp::template get<BoundIndex>(table);
                constexpr std::size_t result_label = BareType<decltype(id)>::target;
                return LabeledBoundID<result_label>{id.template apply<Context>(bounds)};
            }
        )

    private:
        template<typename BoundEv> struct OnBoundEventImpl {
        private:
            static_assert(BoundEv::change != BoundID::EMPTY, "Empty BoundID used in BoundEvent!");
            static_assert(BoundEv::change != BoundID::NONE, "None BoundID used in BoundEvent!");

            struct LBSel {
                template<std::size_t Arg> using Update = typename OnUpdateInfo::OnLBChange::template At<Arg>;
            };
            struct UBSel {
                template<std::size_t Arg> using Update = typename OnUpdateInfo::OnUBChange::template At<Arg>;
            };
            struct BothSel {
                template<std::size_t Arg> using Update = typename OnUpdateInfo::OnBothChange::template At<Arg>;
            };

            using UpdateBoundSelector =
                std::conditional_t<BoundEv::change == BoundID::LB, LBSel,
                std::conditional_t<BoundEv::change == BoundID::UB, UBSel, BothSel>>;

        public:
            using UpdateBounds = typename UpdateBoundSelector::template Update<BoundEv::arg>;
        };

        template<typename Context, typename BoundValueArray, std::size_t F, std::size_t... Rest>
            PropagationResult do_apply_all_bounds(BoundValueArray& bounds, IndexPack<F,Rest...>) const
        {
            if(ivarp::template get<F>(table).template apply<Context>(bounds) == BoundID::EMPTY) {
                return {true};
            }
            return do_apply_all_bounds<Context>(bounds, IndexPack<Rest...>{});
        }

        template<typename Context, typename BoundValueArray>
            PropagationResult do_apply_all_bounds(BoundValueArray& bounds, IndexPack<>) const
        {
            return {false};
        }

    public:
		template<std::size_t Index> IVARP_HD const auto& get() const noexcept {
			return ivarp::template get<Index>(table);
		}
	
        template<typename BoundEv> using UpdateBounds = typename OnBoundEventImpl<BoundEv>::UpdateBounds;
		using BoundIndices = IndexRange<0,num_bounds>;
    };

    template<typename BoundsByArg>
        static inline auto make_runtime_bound_table(BoundsByArg&& argbounds)
    {
        constexpr std::size_t length = TupleSize<BareType<BoundsByArg>>::value;
        using Inds = IndexRange<0,length>;
        auto args = impl::make_rbt_bounds(ivarp::forward<BoundsByArg>(argbounds), Inds{});
        return RuntimeBoundTable<length, decltype(args)>{ivarp::move(args)};
    }
}
