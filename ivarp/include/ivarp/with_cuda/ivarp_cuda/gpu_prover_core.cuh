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
// Created by Phillip Keldenich on 27.02.20.
//

#pragma once

#include "ivarp_cuda/memory.hpp"
#include "ivarp_cuda/error.hpp"
#include "ivarp_cuda/cuda_output_buffer.cuh"
#include "ivarp_cuda/gpu_prover_core_gridsize.cuh"
#include "ivarp_cuda/gpu_propagate.cuh"
#include "ivarp/critical_reducer.hpp"
#include "ivarp/prover_input.hpp"
#include <climits>
#include <vector>
#include <memory>

static inline std::ostream& operator<<(std::ostream& out, dim3 d) {
    return out << "[x: " << d.x << ", y: " << d.y << ", z: " << d.z << "]";
}

namespace ivarp {
namespace impl {
    struct TrivialOnDone {
        template<typename Global> IVARP_HD void operator()(const Global*) const noexcept {}
    };

    /// One copy of this information is held in device memory for each device.
    template<typename ProverInputType_> struct CUDAProverCoreStaticKernelInput {
        static IVARP_HD constexpr std::size_t num_args() noexcept {
            return constexpr_fn<std::size_t, ProverInputType::num_args>();
        }

        using ProverInputType = ProverInputType_;
        using Context = typename ProverInputType::Context;
        using NumberType = typename Context::NumberType;
        using ConcreteNT = typename NumberType::NumberType;
        using RBT = typename ProverInputType::RuntimeBoundTable;
        using RCT = typename ProverInputType::RuntimeConstraintTable;
        using StaticSplitInfo = typename ProverInputType::StaticSplitInfo;
        using QueueElement = ProofDriverQueueEntry<NumberType, num_args()>;
        using Reducer = CriticalReducer<ConcreteNT, num_args(), TrivialOnDone>;
        using RedGlobal = typename Reducer::GlobalMemory;

        ConcreteNT iteration_volume_bound;
        ConcreteNT iteration_single_bound;
        unsigned max_iterations;
        RBT rbt;
        RCT rct;
    };

    struct CUDACriticalHandlerJoinReducer {
        template<typename ReducerType, typename CriticalType>
            IVARP_D void operator()(ReducerType& reducer, const CriticalType& critical)
        {
            reducer.join(critical);
        }
    };

    template<typename OutputType>
    struct CUDACriticalHandlerReport {
        using HostOB = ivarp::cuda::CUDAOutputBuffer<OutputType>;
        using DevOB = typename HostOB::CUDADeviceOutputBuffer;

        template<typename ReducerType, typename CriticalType>
            IVARP_D void operator()(ReducerType& reducer, const CriticalType& critical)
        {
            reducer.count_critical();
            device_output_buffer->push_back(critical);
        }

        DevOB* device_output_buffer;
    };

    template<typename StaticKernelInput, typename CriticalHandler> class CUDAProverCoreKernel {
    public:
        using Context = typename StaticKernelInput::Context;
        using NumberType = typename Context::NumberType;
        using Reducer = typename StaticKernelInput::Reducer;
        using StaticSplitInfo = typename StaticKernelInput::StaticSplitInfo;
        using Cuboid = Array<NumberType, StaticKernelInput::num_args()>;
        using Splitter = ivarp::Splitter<NumberType>;

        static constexpr std::size_t num_args = StaticKernelInput::ProverInputType::num_args;

        IVARP_D CUDAProverCoreKernel(const StaticKernelInput* input, const CriticalHandler& h, Reducer* reducer) :
            input(input), handler(h), reducer(reducer)
        {}

        IVARP_D void run(const Cuboid& element) {
            split_var(element, StaticSplitInfo{});
        }

    private:
        template<std::size_t CurrVar, typename NextSplit, typename... Splits, std::enable_if_t<(CurrVar < NextSplit::arg),int> = 0>
            IVARP_D bool violates_constraints(const Cuboid& element, SplitInfoSequence<NextSplit,Splits...> remaining_splits)
        {
            if(cuda::can_prune<CurrVar, Context>(input->rct, element.data())) {
                return true;
            }
            return violates_constraints<CurrVar+1>(element, remaining_splits);
        }
        template<std::size_t CurrVar, typename NextSplit, typename... Splits, std::enable_if_t<(CurrVar == NextSplit::arg),int> = 0>
            IVARP_D bool violates_constraints(const Cuboid&, SplitInfoSequence<NextSplit,Splits...>)
        {
            return false;
        }

        template<std::size_t CurrVar, std::enable_if_t<(CurrVar < num_args),int> = 0>
            IVARP_D bool violates_constraints(const Cuboid& element, SplitInfoSequence<> remaining_splits)
        {
            if(cuda::can_prune<CurrVar, Context>(input->rct, element.data())) {
                return true;
            }
            return violates_constraints<CurrVar+1>(element, remaining_splits);
        }
        template<std::size_t CurrVar, std::enable_if_t<(CurrVar == num_args),int> = 0>
            IVARP_D bool violates_constraints(const Cuboid&, SplitInfoSequence<>)
        {
            return false;
        }

        template<typename S1, typename... S> IVARP_D void split_var(const Cuboid& element, SplitInfoSequence<S1,S...>)
        {
            using GI = cuda::GridInfoOf<S1, StaticSplitInfo>;
            using SplitEv = BoundEvent<S1::arg, BoundID::BOTH>;
            constexpr std::size_t arg = constexpr_fn<std::size_t,S1::arg>();
            Splitter s(element[arg], constexpr_fn<int,S1::subdivisions>());
            std::size_t leafs = 0, nodes = 0;
            for(int i = GI::initial(); i < constexpr_fn<int,S1::subdivisions>(); i += GI::increment()) {
                ++nodes;
                Cuboid new_c = element;
                new_c[arg] = s.subrange(i);
                if(cuda::propagate<SplitEv,Context>(input->rbt, new_c.data(), 4).empty) {
                    ++leafs;
                    continue;
                }
                if(violates_constraints<S1::arg>(new_c, SplitInfoSequence<S...>{})) {
                    ++leafs;
                    continue;
                }
                split_var(new_c, SplitInfoSequence<S...>{});
            }
            reducer->count_cuboid(nodes);
            reducer->count_leaf(leafs);
        }

        IVARP_D void split_var(const Cuboid& element, SplitInfoSequence<>) {
            reducer->count_leaf();
            handler(*reducer, element);
        }

        const StaticKernelInput* input;
        CriticalHandler handler;
        Reducer* reducer;
    };

    template<typename StaticKernelInput, typename CriticalHandler> void __global__ cuda_prover_core_kernel(
        const StaticKernelInput* static_input,
        const typename StaticKernelInput::QueueElement* elements, ///< gridDim.x elements
        typename StaticKernelInput::RedGlobal* global_mem, ///< An array of at least gridDim.x initialized entries
        CriticalHandler handler ///< A handler for the discovered criticals
    ) {
        using NumberType = typename StaticKernelInput::Context::NumberType;
        using Reducer = typename StaticKernelInput::Reducer;
        extern __shared__ char shared_memory[];
        Reducer reducer(+shared_memory, gridDim.y);
        reducer.initialize_shared();
        CUDAProverCoreKernel<StaticKernelInput, CriticalHandler> kernel(static_input, handler, &reducer);
        Array<NumberType, StaticKernelInput::num_args()> input;
        input.initialize_from(+elements[blockIdx.x].elements);
        kernel.run(input);
        reducer.merge(&global_mem[blockIdx.x]);
    }
}

    template<typename ProverInputType, typename OnCritical> class CUDAProverCore :
        public BasicProverCore<ProverInputType, OnCritical>
    {
    public:
        using Base = BasicProverCore<ProverInputType, OnCritical>;
        using StaticKernelInput = impl::CUDAProverCoreStaticKernelInput<ProverInputType>;
        using Reducer = typename StaticKernelInput::Reducer;
        using ReducerGlobal = typename StaticKernelInput::RedGlobal;
        using Context = typename ProverInputType::Context;
        using NumberType = typename Context::NumberType;
        using ConcreteNT = typename NumberType::NumberType;
        using RBT = typename ProverInputType::RuntimeBoundTable;
        using RCT = typename ProverInputType::RuntimeConstraintTable;
        using StaticSplitInfo = typename ProverInputType::StaticSplitInfo;
        using QueueElement = typename StaticKernelInput::QueueElement;
        using InternalElement = Array<NumberType, StaticKernelInput::num_args()>;
        using CuboidCounts = impl::CuboidCounts;
        using OutputBuffer = ivarp::cuda::CUDAOutputBuffer<InternalElement>;

        static constexpr std::size_t num_args = ProverInputType::num_args;
        static constexpr std::size_t initial_critical_buffer_size = (1u << 20u);
        static constexpr std::size_t maximal_critical_buffer_size = (100u << 20u);

        explicit CUDAProverCore() = default;

        struct PerThreadHostInfo {
            int device_id{-1};
            std::size_t max_grid_dim_x{0};
            std::unique_ptr<OutputBuffer> critical_output_buffer;
            std::unique_ptr<StaticKernelInput, ivarp::cuda::CUDAFreeDeleter> static_kernel_input;
            ivarp::cuda::DeviceArray<ReducerGlobal> reducer_globals;
            ivarp::cuda::DeviceArray<QueueElement> element_buffer;
            std::size_t shared_memory_needed{0};
            int threads_per_block{0};
            int grid_y{1};
            dim3 block_dims{1,1,1};
        };

        void initialize(std::size_t num_threads) {
            per_thread_host_info.resize(num_threads);
        }

        void initialize_per_thread(std::size_t thread_id, std::size_t /*num_threads*/) {
            p_initialize_cuda_device(thread_id);
            per_thread_host_info[thread_id].max_grid_dim_x = this->settings->dequeue_buffer_size;
            p_initialize_static_input(thread_id);
            p_initialize_buffers(thread_id);
            p_initialize_dimensions(thread_id);
        }

        void replace_default_settings(ProverSettings& settings) const {
            settings.thread_count = static_cast<int>(settings.cuda_device_ids.size());
            if(settings.dequeue_buffer_size <= 0) {
                settings.dequeue_buffer_size = 1024;
            }
            if(settings.max_iterations_per_node < 0) {
                settings.max_iterations_per_node = 2;
            }
            if(settings.iteration_single_dimension_criterion < 0) {
                settings.iteration_single_dimension_criterion = 0.66f;
            }
            if(settings.iteration_volume_criterion < 0) {
                settings.iteration_volume_criterion = 0.25f;
            }
        }

        CuboidCounts handle_cuboids_nonfinal(std::size_t thread_id, const std::vector<QueueElement>& input_cuboids,
                                             std::vector<QueueElement>* output_cuboids)
        {
            CuboidCounts result;
            std::size_t beg = 0;
            std::size_t end = input_cuboids.size();
            NonFinal handler{this, output_cuboids, &input_cuboids};
            while(beg < end) {
                result += p_handle_chunk(thread_id, input_cuboids, &beg, end, handler);
            }
            return result;
        }

        CuboidCounts handle_cuboids_final(std::size_t thread_id, const std::vector<QueueElement>& final_elements)
        {
            p_init_output_buffer(thread_id);
            OutputBuffer* obuf = per_thread_host_info[thread_id].critical_output_buffer.get();
            CuboidCounts result;
            Final handler{this, obuf};
            std::size_t beg = 0;
            std::size_t end = final_elements.size();
            while(beg < end) {
                result += p_handle_chunk(thread_id, final_elements, &beg, end, handler);
            }
            return result;
        }

    private:
        struct Final {
            void call_kernel(std::size_t thread_id, std::size_t num_elements,
                             const QueueElement* dev_elms, ReducerGlobal* dev_globs)
            {
                PerThreadHostInfo& info = core->per_thread_host_info[thread_id];
                dim3 grid_dim(static_cast<int>(num_elements), info.grid_y);
                output_buffer->reset();
                impl::CUDACriticalHandlerReport<InternalElement> reporter{output_buffer->pass_to_device()};

                for(;;) {
                    impl::cuda_prover_core_kernel<<<grid_dim, info.block_dims, info.shared_memory_needed>>>(
                        info.static_kernel_input.get(), dev_elms, dev_globs, reporter
                    );
                    throw_if_cuda_error("Synchronous launch error", cudaPeekAtLastError());
                    throw_if_cuda_error("Asynchronous launch error", cudaDeviceSynchronize());
                    for(const InternalElement& critical : output_buffer->get_output()) {
                        (*core->on_critical)(Context{}, critical);
                    }
                    if(output_buffer->buffer_was_sufficient()) {
                        break;
                    }
                    output_buffer->prepare_next_call();
                }
            }

            void handle_result_bounds(const ReducerGlobal* /*globals*/, std::size_t /*num*/) const noexcept {}

            CUDAProverCore* core;
            OutputBuffer* output_buffer;
        };

        struct NonFinal {
            void call_kernel(std::size_t thread_id, std::size_t num_elements,
                             const QueueElement* dev_elms, ReducerGlobal* dev_globs)
            {
                PerThreadHostInfo& info = core->per_thread_host_info[thread_id];
                dim3 grid_dim(static_cast<int>(num_elements), info.grid_y);
                impl::CUDACriticalHandlerJoinReducer reduce;
                impl::cuda_prover_core_kernel<<<grid_dim, info.block_dims, info.shared_memory_needed>>>(
                    info.static_kernel_input.get(), dev_elms, dev_globs, reduce
                );
                throw_if_cuda_error("Synchronous launch error", cudaPeekAtLastError());
                throw_if_cuda_error("Asynchronous launch error", cudaDeviceSynchronize());
            }

            void handle_result_bounds(const ReducerGlobal* globals, std::size_t num) {
                for(std::size_t i = 0; i < num; ++i) {
                    if(!globals[i].empty) {
                        output->emplace_back();
                        QueueElement& e = output->back();
                        e.depth = (*input)[index].depth;
                        std::copy_n(globals[i].bounds.begin(), num_args, +e.elements);
                    }
                    ++index;
                }
            }

            CUDAProverCore* core;
            std::vector<QueueElement>* output;
            const std::vector<QueueElement>* input;
            std::size_t index{0};
        };

        IVARP_H void p_initialize_dimensions(std::size_t thread_id) {
            PerThreadHostInfo& info = per_thread_host_info[thread_id];
            info.grid_y = impl::cuda::get_grid_y(StaticSplitInfo{});
            info.block_dims = impl::cuda::get_block_dims(info.device_id, this->settings->cuda_threads_per_block,
                                                         StaticSplitInfo{});
            info.threads_per_block = info.block_dims.x * info.block_dims.y * info.block_dims.z;
            info.shared_memory_needed =
                Reducer::shared_memory_size_needed(static_cast<std::size_t>(info.threads_per_block));
        }

        template<typename Handler>
        IVARP_H CuboidCounts p_handle_chunk(std::size_t thread_id, const std::vector<QueueElement>& in_elements,
                                            std::size_t* beg, std::size_t end, Handler& h)
        {
            for(;;) {
                try {
                    return p_try_call_kernel(thread_id, in_elements, beg, end, h);
                } catch(const CUDAError& err) {
                    // Check whether this is one of the errors that we can prevent by moving to
                    // fewer blocks (i.e., watchdog or configuration issues as OOM or watchdog timer)
                    if(err.get_error_code() == cudaErrorInvalidConfiguration ||
                       err.get_error_code() == cudaErrorLaunchTimeout)
                    {
                        PerThreadHostInfo& info = per_thread_host_info[thread_id];
                        if(info.max_grid_dim_x > 4) {
                            info.max_grid_dim_x /= 2;
                            continue;
                        }
                    }
                    throw;
                }
            }
        }

        template<typename Handler>
        IVARP_H CuboidCounts p_try_call_kernel(std::size_t thread_id, const std::vector<QueueElement>& in_elements,
                                               std::size_t* beg, std::size_t end, Handler& h)
        {
            PerThreadHostInfo& info = per_thread_host_info[thread_id];
            std::size_t num = end - *beg;
            if(num > info.max_grid_dim_x) {
                num = info.max_grid_dim_x;
            }
            p_prepare_args(thread_id, num, in_elements.begin() + *beg);
            h.call_kernel(thread_id, num,
                          info.element_buffer.pass_to_device_copy_n(num),
                          info.reducer_globals.pass_to_device_copy_n(num));
            info.reducer_globals.read_n_from_device(num);
            h.handle_result_bounds(info.reducer_globals.begin(), num);
            *beg += num;
            return p_handle_result_counts(info.reducer_globals.begin(), num);
        }

        template<typename Iter>
        IVARP_H void p_prepare_args(std::size_t thread_id, std::size_t num, Iter elements_begin) {
            PerThreadHostInfo& info = per_thread_host_info[thread_id];
            auto elements_end = elements_begin + num;
            std::copy(elements_begin, elements_end, info.element_buffer.begin());
            for(std::size_t i = 0; i < num; ++i) {
                info.reducer_globals[i] = GlobalMemoryInit{};
            }
        }

        IVARP_H CuboidCounts p_handle_result_counts(const ReducerGlobal* globals, std::size_t num) {
            CuboidCounts result;
            for(std::size_t i = 0; i < num; ++i) {
                result += globals[i].counts();
            }
            return result;
        }

        IVARP_H void p_initialize_cuda_device(std::size_t cpu_thread_id) {
            int device_id = this->settings->cuda_device_ids.at(cpu_thread_id);
            per_thread_host_info[cpu_thread_id].device_id = device_id;

            const auto sh_bank_size = sizeof(ConcreteNT) * CHAR_BIT == 64 ? cudaSharedMemBankSizeEightByte :
                                      sizeof(ConcreteNT) * CHAR_BIT == 32 ? cudaSharedMemBankSizeFourByte :
                                                                            cudaSharedMemBankSizeDefault;

            throw_if_cuda_error("Could not set the current CUDA device!", cudaSetDevice(device_id));
            warn_if_cuda_error("Could not set the device heap size to 0!",
                               cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0));
            warn_if_cuda_error("Could not set the device printf buffer size to 0!",
                               cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0));
            warn_if_cuda_error("Could not set the shared memory bank size!",
                               cudaDeviceSetSharedMemConfig(sh_bank_size));
        }

        IVARP_H void p_initialize_static_input(std::size_t thread_id) {
            StaticKernelInput input{
                convert_number<ConcreteNT>(this->settings->iteration_volume_criterion),
                convert_number<ConcreteNT>(this->settings->iteration_single_dimension_criterion),
                static_cast<unsigned>(this->settings->max_iterations_per_node),
                *this->rbt, *this->rct
            };
            void* target;
            throw_if_cuda_error("Could not allocate static kernel input memory!",
                                cudaMalloc(&target, sizeof(StaticKernelInput)));
            throw_if_cuda_error("Could not copy static kernel input to device!",
                                cudaMemcpy(target, static_cast<void*>(&input), sizeof(StaticKernelInput), cudaMemcpyHostToDevice));
            per_thread_host_info[thread_id].static_kernel_input.reset(static_cast<StaticKernelInput*>(target));
        }

        IVARP_H void p_initialize_buffers(std::size_t thread_id) {
            std::size_t maxlen = per_thread_host_info[thread_id].max_grid_dim_x;
            PerThreadHostInfo& info = per_thread_host_info[thread_id];
            info.reducer_globals = cuda::DeviceArray<ReducerGlobal>{maxlen};
            info.element_buffer = cuda::DeviceArray<QueueElement>{maxlen};
        }

        IVARP_H void p_init_output_buffer(std::size_t thread_id) {
            PerThreadHostInfo& info = per_thread_host_info[thread_id];
            if(!info.critical_output_buffer) {
                info.critical_output_buffer =
                    std::make_unique<OutputBuffer>(initial_critical_buffer_size, maximal_critical_buffer_size);
            }
        }

        std::vector<PerThreadHostInfo> per_thread_host_info;
    };

    template<typename ProverInputType, typename OnCritical> constexpr std::size_t CUDAProverCore<ProverInputType, OnCritical>::num_args;
    template<typename ProverInputType, typename OnCritical> constexpr std::size_t CUDAProverCore<ProverInputType, OnCritical>::initial_critical_buffer_size;
    template<typename ProverInputType, typename OnCritical> constexpr std::size_t CUDAProverCore<ProverInputType, OnCritical>::maximal_critical_buffer_size;
}
