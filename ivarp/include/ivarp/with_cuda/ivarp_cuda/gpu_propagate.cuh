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
// Created by Phillip Keldenich on 02.03.20.
//

#pragma once

namespace ivarp {
namespace impl {
namespace cuda {
    template<std::size_t... Triggered> static inline IVARP_D
        void trigger(bool* triggered, IndexPack<Triggered...>) noexcept
    {
        ConstructWithAny{(triggered[Triggered] = true)...};
    }

    template<std::size_t Length> static inline IVARP_D void clear_triggered(bool* triggered) noexcept {
        for(std::size_t i = 0; i < Length; ++i) {
            triggered[i] = false;
        }
    }

    template<typename Context, std::size_t BoundIndex, std::size_t BoundCount, typename RBT,
             std::enable_if_t<(BoundIndex == BoundCount), int> = 0> static inline IVARP_D
        PropagationResult propagation_round_apply_bound(const RBT& runtime_bounds,
                                                        typename Context::NumberType* apply_to,
                                                        const bool* trigger_in, bool* trigger_out, bool* stable) noexcept
    {
        return PropagationResult{false};
    }

    template<typename Context, std::size_t BoundIndex, std::size_t BoundCount,
             typename RBT, std::enable_if_t<(BoundIndex < BoundCount), int> = 0> static inline IVARP_D
        PropagationResult propagation_round_apply_bound(const RBT& runtime_bounds,
                                                        typename Context::NumberType* apply_to,
                                                        const bool* trigger_in, bool* trigger_out, bool* stable) noexcept
    {
        if(trigger_in[BoundIndex]) {
            auto result = runtime_bounds.template apply_bound<BoundIndex, Context>(apply_to);
            using BothUpdateEv = BoundEvent<decltype(result)::target, BoundID::BOTH>;
            using UBUpdateEv = BoundEvent<decltype(result)::target, BoundID::UB>;
            using LBUpdateEv = BoundEvent<decltype(result)::target, BoundID::LB>;
            using BothUp = typename RBT::template UpdateBounds<BothUpdateEv>;
            using UBUp = typename RBT::template UpdateBounds<UBUpdateEv>;
            using LBUp = typename RBT::template UpdateBounds<LBUpdateEv>;

            switch(result.bound_id) {
                case BoundID::EMPTY:
                    return PropagationResult{true};
                case BoundID::NONE:
                default:
                    break;

                case BoundID::LB:
                    *stable = false;
                    trigger(trigger_out, LBUp{});
                    break;

                case BoundID::UB:
                    *stable = false;
                    trigger(trigger_out, UBUp{});
                    break;

                case BoundID::BOTH:
                    *stable = false;
                    trigger(trigger_out, BothUp{});
                    break;
            }
        }
        return propagation_round_apply_bound<Context, BoundIndex+1, BoundCount>(
            runtime_bounds, apply_to, trigger_in, trigger_out, stable
        );
    }

    template<typename Context, typename RBT> static inline IVARP_D
    PropagationResult propagation_round(const RBT& runtime_bounds, typename Context::NumberType* apply_to,
                                        const bool* trigger_in, bool* trigger_out, bool* stable) noexcept
    {
        clear_triggered<RBT::num_bounds>(trigger_out);
        return propagation_round_apply_bound<Context, 0, RBT::num_bounds>(
            runtime_bounds, apply_to, trigger_in, trigger_out, stable
        );
    }

    template<typename BoundEv, typename Context, typename RBT> static inline IVARP_D
    PropagationResult propagate(const RBT& runtime_bounds, typename Context::NumberType* apply_to, std::size_t it_limit)
    {
        using Update = typename RBT::template UpdateBounds<BoundEv>;
        bool triggered1[RBT::num_bounds];
        bool triggered2[RBT::num_bounds];
        bool *t1 = triggered1;
        bool *t2 = triggered2;
        bool stable = true;
        clear_triggered<RBT::num_bounds>(triggered1);
        trigger(triggered1, Update{});

        for(std::size_t i = 0; i < it_limit; ++i) {
            if(propagation_round<Context>(runtime_bounds, apply_to, t1, t2, &stable).empty) {
                return PropagationResult{true};
            }
            if(stable) {
                break;
            }
            stable = true;
            ivarp::ivswap(t1, t2);
        }
        return PropagationResult{false};
    }

    template<typename Context, typename RCTRow> static inline IVARP_D
        bool can_prune_row(const RCTRow&, const typename Context::NumberType*, IndexPack<>)
    {
        return false;
    }

    template<typename Context, typename RCTRow, std::size_t I1, std::size_t... Inds> static inline IVARP_D
        bool can_prune_row(const RCTRow& row, const typename Context::NumberType* apply_to, IndexPack<I1,Inds...>)
    {
        if(!possibly(ivarp::get<I1>(row).template array_evaluate<Context>(apply_to))) {
            return true;
        }
        return can_prune_row<Context>(row, apply_to, IndexPack<Inds...>{});
    }

    template<std::size_t NewSplitVar, typename Context, typename RCT> static inline IVARP_D
        bool can_prune(const RCT& runtime_constraints, const typename Context::NumberType* apply_to)
    {
        const auto& constraints = ivarp::get<NewSplitVar>(runtime_constraints);
        return can_prune_row<Context>(constraints, apply_to, TupleIndexPack<BareType<decltype(constraints)>>{});
    }
}
}
}
