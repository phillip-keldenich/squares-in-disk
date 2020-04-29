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
// Created by Phillip Keldenich on 14.02.20.
//

#pragma once

#include "bound_event.hpp"
#include "propagation_result.hpp"

/**
 * \file iterated_recursive.hpp
 * \brief Implements a (CPU-focused) bound propagation algorithm that works in several iterations.
 */

namespace ivarp {
namespace propagate_iterated_recursive {
namespace impl {
    template<std::size_t NumBounds> struct IteratedRecursivePropInfo {
        static constexpr std::size_t num_bounds = NumBounds;

        IteratedRecursivePropInfo() noexcept { // NOLINT
            clear();
        }

        void clear() noexcept {
            std::fill_n(+rec_count, num_bounds, std::uint8_t{0});
            std::fill_n(+unstable, num_bounds, std::uint8_t{0});
            num_unstable = 0;
        }

        std::uint8_t rec_count[num_bounds];
        std::uint8_t unstable[num_bounds];
        std::size_t num_unstable;
        std::size_t unstable_bounds[num_bounds];
    };

    template<typename Context, typename RBT> class Propagator {
    public:
        static constexpr std::size_t num_bounds = RBT::num_bounds;
        using NumberType = typename Context::NumberType;
        using DBA = DynamicBoundApplication<RBT, Context>;
        using Queue = IndexQueueSet<num_bounds>;
        using Info = IteratedRecursivePropInfo<num_bounds>;

        template<typename InitialIndexPack> explicit Propagator(InitialIndexPack, const RBT& runtime_bounds,
                                                                const DBA& dba) :
            rbt(&runtime_bounds), dba(&dba), queue(), info()
        {
            queue.enqueue_all(InitialIndexPack{});
        }

        PropagationResult run(NumberType* values, std::size_t it_limit, std::uint8_t rec_limit) {
            for(std::size_t i = 0; i < it_limit; ++i) {
                while(!queue.empty()) {
                    std::size_t next_bound = queue.dequeue();
                    if(dba->apply_bound_with_info(next_bound, values, queue, rec_limit, info).empty) {
                        return {true};
                    }
                }

                if(info.num_unstable > 0) {
                    queue.enqueue_range(+info.unstable_bounds, info.unstable_bounds+info.num_unstable);
                    info.clear();
                } else {
                    break;
                }
            }
            return {false};
        }

    private:
        const RBT* const rbt;
        const DBA* const dba;
        Queue queue;
        Info info;
    };

    template<typename Update, typename Context, typename RBT>
        static inline PropagationResult do_propagate(const RBT& runtime_bounds, typename Context::NumberType* apply_to,
                                                     const DynamicBoundApplication<RBT, Context>& dba, std::size_t it_limit, std::uint8_t rec_limit)
    {
        impl::Propagator<Context, RBT> propagator{Update{}, runtime_bounds, dba};
        return propagator.run(apply_to, it_limit, rec_limit);
    }

    template<typename Update, typename Context, typename RBT, std::enable_if_t<!std::is_same<Update,IndexPack<>>::value,int> = 0>
        static inline PropagationResult do_propagate(const RBT& runtime_bounds, typename Context::NumberType* apply_to,
                                                     const DynamicBoundApplication<RBT, Context>& dba, std::size_t it_limit, std::uint8_t rec_limit)
    {
        impl::Propagator<Context, RBT> propagator{Update{}, runtime_bounds, dba};
        return propagator.run(apply_to, it_limit, rec_limit);
    }

    template<typename Update, typename Context, typename RBT, std::enable_if_t<std::is_same<Update,IndexPack<>>::value,int> = 0>
        static inline PropagationResult do_propagate(const RBT&, typename Context::NumberType*,
                                                     const DynamicBoundApplication<RBT, Context>&, std::size_t, std::uint8_t)
    {
        return PropagationResult{false};
    }
}

    template<typename BoundEv, typename Context, typename RBT>
        static inline PropagationResult propagate(const RBT& runtime_bounds, typename Context::NumberType* apply_to,
                                                  const DynamicBoundApplication<RBT, Context>& dba,
                                                  std::size_t it_limit, std::uint8_t rec_limit)
    {
        using Update = typename RBT::template UpdateBounds<BoundEv>;
        impl::Propagator<Context, RBT> propagator{Update{}, runtime_bounds, dba};
        return propagator.run(apply_to, it_limit, rec_limit);
    }
}
}
