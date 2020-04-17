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
// Created by Phillip Keldenich on 11.12.19.
//

#pragma once

#include "ivarp/cuda.hpp"
#include "ivarp/number.hpp"
#include "ivarp/array.hpp"

namespace ivarp {
    struct GlobalMemoryInit {};

    template<typename BoundType, std::size_t NumVars, typename OnDone> class CriticalReducer {
        static_assert(std::is_trivial<BoundType>::value, "BoundType must be trivial!");

    public:
        using CuboidCounts = impl::CuboidCounts;
        using IntervalType = Interval<BoundType>;

        struct GlobalMemory {
            static_assert(std::is_trivially_destructible<CuboidCounts>::value, "Counts should be trivial to destroy!");

            GlobalMemory() = default;
            IVARP_HD explicit GlobalMemory(GlobalMemoryInit) noexcept :
                blocks_done(0), lock(0), empty{true}
            {
                bounds.fill(IntervalType(0,0));
                new (+ccbuf.buffer) CuboidCounts();
            }

            IVARP_HD GlobalMemory& operator=(GlobalMemoryInit) noexcept {
                blocks_done = 0;
                lock = 0;
                bounds.fill(IntervalType(0,0));
                empty = true;
                new (+ccbuf.buffer) CuboidCounts();
                return *this;
            }

            IVARP_HD CuboidCounts& counts() noexcept {
                return *reinterpret_cast<CuboidCounts*>(+ccbuf.buffer);
            }

            IVARP_HD const CuboidCounts& counts() const noexcept {
                return *reinterpret_cast<const CuboidCounts*>(+ccbuf.buffer);
            }

            struct alignas(alignof(CuboidCounts)) CCBuf {
                char buffer[sizeof(CuboidCounts)];
            };

            CCBuf ccbuf;
            unsigned blocks_done;
            int lock;
            Array<IntervalType, NumVars> bounds;
            bool empty;
        };

        IVARP_HD static constexpr std::size_t shared_memory_size_needed(std::size_t threads_per_block) noexcept {
            return 2 * sizeof(BoundType) * NumVars * threads_per_block + threads_per_block * sizeof(CuboidCounts);
        }

#if defined(__CUDA_ARCH__)
        IVARP_D void initialize_shared() noexcept {
            for(std::size_t i = 0; i < NumVars; ++i) {
                m_shared_lb[i] = impl::inf_value<BoundType>();
                m_shared_ub[i] = -impl::inf_value<BoundType>();
            }
            new (m_shared_counts) CuboidCounts();
        }

        IVARP_D void count_cuboid(unsigned num = 1) noexcept {
            m_shared_counts->num_cuboids += num;
        }

        IVARP_D void count_leaf(unsigned num = 1) noexcept {
            m_shared_counts->num_leaf_cuboids += num;
        }

        IVARP_D void count_critical(unsigned num = 1) noexcept {
            m_shared_counts->num_critical_cuboids += num;
        }

        IVARP_D void count_repetition(unsigned num = 1) noexcept {
            m_shared_counts->num_repeated_nodes += num;
        }

        template<typename... OnDoneArgs>
        IVARP_D CriticalReducer(char* shared_memory, dim3 grid_dim, OnDoneArgs&&... on_done_args) noexcept :
            block_thread_id(threadIdx.z * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x),
            threads_per_block(blockDim.x * blockDim.y * blockDim.z),
            num_blocks(grid_dim.x * grid_dim.y),
            m_shared_lb(reinterpret_cast<BoundType*>(shared_memory + shared_lb_offset(block_thread_id))),
            m_shared_ub(reinterpret_cast<BoundType*>(shared_memory + shared_ub_offset(block_thread_id, threads_per_block))),
            m_shared_counts(reinterpret_cast<CuboidCounts*>(shared_memory + shared_count_offset(block_thread_id, threads_per_block))),
            m_on_done(std::forward<OnDoneArgs>(on_done_args)...)
        {}

        IVARP_D void merge(GlobalMemory* g) noexcept {
            __syncthreads();
            if(block_thread_id == 0) {
                Array<BoundType, NumVars> lbs;
                Array<BoundType, NumVars> ubs;
                merge_handle_bounds(lbs, ubs);
                CuboidCounts counts = merge_handle_counts();
                merge_global(g, lbs, ubs, counts);
            }
        }

        private:
        IVARP_D void merge_handle_bounds(Array<BoundType, NumVars>& lbs, Array<BoundType, NumVars>& ubs) {
            for(std::size_t i = 0; i < NumVars; ++i) {
                lbs[i] = m_shared_lb[i];
                ubs[i] = m_shared_ub[i];
                for(std::size_t j = i + NumVars; j < threads_per_block * NumVars; j += NumVars) {
                    BoundType l = m_shared_lb[j];
                    if(l < lbs[i]) {
                        lbs[i] = l;
                    }
                    BoundType u = m_shared_ub[j];
                    if(u > ubs[i]) {
                        ubs[i] = u;
                    }
                }
            }
        }

        IVARP_D CuboidCounts merge_handle_counts() {
            CuboidCounts counts = m_shared_counts[0];
            for(std::size_t i = 1; i < threads_per_block; ++i) {
                counts += m_shared_counts[i];
            }
            return counts;
        }

        IVARP_D static bool bounds_are_empty(const Array<BoundType,NumVars>& lbs,
                                             const Array<BoundType,NumVars>& ubs) noexcept
        {
            for(std::size_t i = 0; i < NumVars; ++i) {
                if(lbs[i] > ubs[i]) {
                    return true;
                }
            }
            return false;
        }

        IVARP_D void block_done(GlobalMemory* g) noexcept {
            if(atomicAdd(&g->blocks_done, 1u) == num_blocks-1) {
                // we are the last block that's done
                m_on_done(g);
            }
        }

        IVARP_D void merge_global_bounds(GlobalMemory* g, const Array<BoundType,NumVars>& lbs,
                                         const Array<BoundType,NumVars>& ubs)
        {
            // take the lock; only one thread per block runs this.
            while(atomicCAS(&g->lock, 0, 1) != 0);

            if(g->empty) {
                g->empty = false;
                for(std::size_t i = 0; i < NumVars; ++i) {
                    g->bounds[i] = IntervalType{lbs[i],ubs[i]};
                }
            } else {
                for(std::size_t i = 0; i < NumVars; ++i) {
                    g->bounds[i].do_join(IntervalType{lbs[i],ubs[i]});
                }
            }

            // release the lock
            g->lock = 0;
        }

        IVARP_D void merge_global(GlobalMemory* g, const Array<BoundType,NumVars>& lbs,
                                  const Array<BoundType,NumVars>& ubs, const CuboidCounts& counts) noexcept
        {
            // handle counts
            g->counts().gpu_atomic_add(counts);

            // check whether we are empty.
            if(!bounds_are_empty(lbs, ubs)) {
                // handle the actual merge
                merge_global_bounds(g, lbs, ubs);
            }

            // we are done
            block_done(g);
        }
        public:
#endif

        template<typename... OnDoneArgs>
        IVARP_H CriticalReducer(char* shared_memory, std::size_t thread_id, std::size_t num_threads, OnDoneArgs&&... on_done_args) noexcept :
            block_thread_id(thread_id), threads_per_block(num_threads), num_blocks(1),
            m_shared_lb(reinterpret_cast<BoundType*>(shared_memory + shared_lb_offset(thread_id))),
            m_shared_ub(reinterpret_cast<BoundType*>(shared_memory + shared_ub_offset(thread_id, threads_per_block))),
            m_shared_counts(reinterpret_cast<CuboidCounts*>(shared_memory + shared_count_offset(thread_id, threads_per_block))),
            m_on_done(std::forward<OnDoneArgs>(on_done_args)...)
        {}

        template<typename ArrayType>
        IVARP_HD void join(const ArrayType& critical) noexcept {
            for(std::size_t i = 0; i < NumVars; ++i) {
                const IntervalType& t = critical[i];
                BoundType l = m_shared_lb[i];
                if (l > t.lb()) {
                    m_shared_lb[i] = t.lb();
                }
                BoundType u = m_shared_ub[i];
                if (u < t.ub()) {
                    m_shared_ub[i] = t.ub();
                }
            }
        }

    private:
        IVARP_HD static constexpr std::size_t shared_lb_offset(std::size_t local_tid) noexcept {
            return local_tid * NumVars * sizeof(BoundType);
        }

        IVARP_HD static constexpr std::size_t shared_ub_offset(std::size_t local_tid,
                                                               std::size_t threads_per_block) noexcept
        {
            return (threads_per_block + local_tid) * NumVars * sizeof(BoundType);
        }

        IVARP_HD static constexpr std::size_t shared_count_offset(std::size_t local_tid,
                                                                  std::size_t threads_per_block) noexcept
        {
            return 2 * sizeof(BoundType) * NumVars * threads_per_block + local_tid * sizeof(CuboidCounts);
        }

        const std::size_t block_thread_id, threads_per_block, num_blocks;
        BoundType* m_shared_lb;
        BoundType* m_shared_ub;
        CuboidCounts* m_shared_counts;
        OnDone m_on_done;
    };
}
