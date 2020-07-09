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

#include "ivarp/cpu_prover/cpu_prover_core.hpp"

namespace ivarp {
namespace impl {
    template<typename ProverInputType, typename OnCritical, typename ProgressReporter>
    static inline IVARP_H bool cpu_run_prover(const ProverInputType& input,
                                              const OnCritical& c,
                                              ProofInformation& info,
                                              ProverSettings& settings,
                                              ProgressReporter& progress)
    {
        using Core = CPUProverCore<ProverInputType, OnCritical>;
        using Driver = ProofDriver<ProverInputType, Core, OnCritical, ProgressReporter>;
        Core core;
        core.replace_default_settings(settings);
        Driver driver{input, &c, &progress, &info, &core, settings};
        return driver.run();
    }
}
}

