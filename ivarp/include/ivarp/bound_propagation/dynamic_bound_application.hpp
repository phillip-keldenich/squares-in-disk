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
// Created by Phillip Keldenich on 17.02.20.
//

#pragma once

namespace ivarp {
namespace propagate_iterated_recursive {
namespace impl {
    template<std::size_t NumBounds> struct IteratedRecursivePropInfo;
}
}

    struct DynamicBoundEvent {
        std::size_t target;
        BoundID   bound_id;
    };

    template<typename RuntimeBoundTable, typename Context>
        class DynamicBoundApplication
    {
        using NumberType = typename Context::NumberType;
        static constexpr std::size_t num_bounds = RuntimeBoundTable::num_bounds;
        using RecPropInfo = propagate_iterated_recursive::impl::IteratedRecursivePropInfo<num_bounds>;
        using Queue = IndexQueueSet<num_bounds>;

        class BoundApplierBase {
        public:
            virtual ~BoundApplierBase() = default;
            virtual DynamicBoundEvent apply_bound(NumberType* apply_to) const = 0;
            virtual PropagationResult apply_bound_with_info(NumberType* apply_to, Queue& bound_queue,
                                                            RecPropInfo & info, std::uint8_t limit) const = 0;
        };

        using ApplierPtr = std::unique_ptr<const BoundApplierBase>;

        template<std::size_t BoundIndex> class BoundApplier : public BoundApplierBase {
        public:
            explicit BoundApplier(DynamicBoundApplication* outer_this) :
                rbt(outer_this->runtime_bounds)
            {}

            DynamicBoundEvent apply_bound(NumberType* apply_to) const override {
                auto labeled_result = rbt->template apply_bound<BoundIndex, Context>(apply_to);
                return DynamicBoundEvent{decltype(labeled_result)::target, labeled_result.bound_id};
            }

            PropagationResult apply_bound_with_info(NumberType* apply_to, Queue& bound_queue,
                                                    RecPropInfo& info, std::uint8_t limit) const override
            {
                auto labeled_result = rbt->template apply_bound<BoundIndex, Context>(apply_to);
                constexpr std::size_t target = decltype(labeled_result)::target;
                using UpdateLB =
                    typename RuntimeBoundTable::template UpdateBounds<impl::BoundEvent<target, BoundID::LB>>;
                using UpdateUB =
                    typename RuntimeBoundTable::template UpdateBounds<impl::BoundEvent<target, BoundID::UB>>;
                using UpdateBB =
                    typename RuntimeBoundTable::template UpdateBounds<impl::BoundEvent<target, BoundID::BOTH>>;

                switch(labeled_result.bound_id) {
                    case BoundID::EMPTY:
                        return {true};

                    case BoundID::LB:
                        p_handle_recursion(bound_queue, info, limit, UpdateLB{});
                        break;

                    case BoundID::UB:
                        p_handle_recursion(bound_queue, info, limit, UpdateUB{});
                        break;

                    case BoundID::BOTH:
                        p_handle_recursion(bound_queue, info, limit, UpdateBB{});
                        break;

                    default: break;
                }
                return {false};
            }

        private:
            template<std::size_t... BoundsUpdated>
            void p_handle_recursion(Queue& bound_queue, RecPropInfo& info, std::uint8_t limit,
                                    IndexPack<BoundsUpdated...>) const
            {
                std::size_t arr[] = {BoundsUpdated...};
                for(std::size_t a : arr) {
                    if(info.rec_count[a] >= limit) {
                        if(!bound_queue.is_present(a)) {
                            if(!info.unstable[a]) {
                                info.unstable_bounds[info.num_unstable++] = a;
                                info.unstable[a] = std::uint8_t{1};
                            }
                        }
                    } else {
                        if(bound_queue.enqueue(a)) {
                            info.rec_count[a] += 1;
                        }
                    }
                }
            }

            void p_handle_recursion(Queue&, RecPropInfo&,std::uint8_t,IndexPack<>) const noexcept {}

            const RuntimeBoundTable* const rbt;
        };

        template<std::size_t... BoundInds> void p_generate_all_appliers(IndexPack<BoundInds...>) {
            ConstructWithAny{((appliers[BoundInds] = std::make_unique<BoundApplier<BoundInds>>(this)), 0)...};
        }

    public:
        explicit DynamicBoundApplication(const RuntimeBoundTable* runtime_bounds) :
            runtime_bounds(runtime_bounds)
        {
            p_generate_all_appliers(IndexRange<0,num_bounds>{});
        }

        DynamicBoundEvent apply_bound(std::size_t bound, NumberType* values) const {
            return appliers[bound]->apply_bound(values);
        }

        PropagationResult apply_bound_with_info(std::size_t bound, NumberType* apply_to, Queue& q,
                                                std::uint8_t rec_limit, RecPropInfo& info) const
        {
            return appliers[bound]->apply_bound_with_info(apply_to,  q, info, rec_limit);
        }

    private:
        const RuntimeBoundTable* runtime_bounds;
        ApplierPtr appliers[num_bounds];
    };
}
