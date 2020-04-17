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

#pragma once

#include <thread>
#include <memory>
#include <atomic>

#include "ivarp/cuda.hpp"
#include "ivarp/metaprogramming.hpp"
#include "ivarp/prover_input.hpp"
#include "ivarp/prover/proof_information.hpp"
#include "ivarp/default_progress_observer.hpp"
#include "ivarp/cuda_device_discovery.hpp"
#include "ivarp/prover/proof_driver.hpp"
#include "ivarp/prover/basic_prover_core.hpp"
#include "ivarp/prover/cpu_run_prover.hpp"
#ifdef __CUDACC__
#include "ivarp_cuda/gpu_run_prover.cuh"
#endif

namespace ivarp {
namespace impl {
    static inline ProverSettings replace_default_settings(ProverSettings s) {
        if (s.cuda_device_ids.size() == 1 && s.cuda_device_ids.front() < 0) {
            s.cuda_device_ids = viable_cuda_device_ids();
        }

        if (!s.cuda_device_ids.empty()) {
            s.thread_count = static_cast<int>(s.cuda_device_ids.size());
        }

        if (s.thread_count <= 0) {
            if (s.cuda_device_ids.empty()) {
                s.thread_count = static_cast<int>(std::thread::hardware_concurrency());
            } else {
                s.thread_count = static_cast<int>(s.cuda_device_ids.size());
            }
        }

        return s;
    }
}

    template<typename ProverInputType, typename ProgressObserver, typename OnCritical>
    static inline IVARP_H bool run_prover(const ProverInputType& input, const OnCritical& c,
                                          ProofInformation* info, const ProverSettings& s, 
                                          ProgressObserver progress)
    {
        ProofInformation info_out;
        if (info == nullptr) {
            info = &info_out;
        }

        ProverSettings actual_settings = impl::replace_default_settings(s);
#ifdef __CUDACC__
        if (!actual_settings.cuda_device_ids.empty()) {
            return impl::gpu_run_prover(input, c, *info, actual_settings, progress);
        }
#endif
        return impl::cpu_run_prover(input, c, *info, actual_settings, progress);
    }

    template<typename ProverInputType, typename OnCritical>
    static inline IVARP_H bool run_prover(const ProverInputType& input, const OnCritical& c,
                                          ProofInformation* info = nullptr, const ProverSettings& s = {})
    {
        return run_prover(input, c, info, s, DefaultProgressObserver{});
    }

    template<typename ProverInputType, typename OnCritical>
    static inline IVARP_H bool run_prover(const ProverInputType& input, const OnCritical& c,
                                          const ProverSettings& s)
    {
        return run_prover(input, c, nullptr, s, DefaultProgressObserver{});
    }
}
