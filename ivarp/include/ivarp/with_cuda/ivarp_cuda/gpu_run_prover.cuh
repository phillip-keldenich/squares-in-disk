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

#include "ivarp_cuda/gpu_prover_core.cuh"
#include "ivarp/critical_reducer.hpp"

namespace ivarp {
namespace impl {
    template<typename PIT> struct HasStaticallySplitVariables : std::integral_constant<bool,
        !std::is_same<typename PIT::StaticSplitInfo, SplitInfoSequence<>>::value>
    {};

    template<typename PIT> struct IsCUDAProofInput : std::integral_constant<bool,
        !std::is_same<typename PIT::StaticSplitInfo, SplitInfoSequence<>>::value &&
        IsCUDAProofContext<typename PIT::Context>::value>
    {};

    template<typename ProverInputType, typename OnCritical, typename ProgressReporter,
             std::enable_if_t<IsCUDAProofInput<ProverInputType>::value,int> = 0>
    static inline IVARP_H bool gpu_run_prover(const ProverInputType& input,
                                              const OnCritical& c,
                                              ProofInformation& info,
                                              ProverSettings& settings,
                                              ProgressReporter& progress)
    {
        using Core = CUDAProverCore<ProverInputType, OnCritical>;
        using Driver = ProofDriver<ProverInputType, Core, OnCritical, ProgressReporter>;
        Core core;
        core.replace_default_settings(settings);
        Driver driver{input, &c, &progress, &info, &core, settings};
        return driver.run();
    }

    template<typename ProverInputType, typename OnCritical, typename ProgressReporter,
             std::enable_if_t<!IsCUDAProofInput<ProverInputType>::value, int> = 0>
    static inline IVARP_H bool gpu_run_prover(const ProverInputType& input,
                                              const OnCritical& c,
                                              ProofInformation& info,
                                              ProverSettings& settings,
                                              ProgressReporter& progress)
    {
        return cpu_run_prover(input, c, info, settings, progress);
    }
}
}
