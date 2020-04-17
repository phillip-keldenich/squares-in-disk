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
#include <doctest/doctest_fixed.hpp>
#include "../test_util.hpp"
#include "ivarp/run_prover.hpp"

using namespace ivarp;

namespace {
    template<typename F>
    struct OnDone {
        IVARP_HD explicit OnDone(Array<Interval<F>,3>* o) noexcept :
            output(o)
        {}

        template<typename Global> IVARP_HD void operator()(const Global* g) const noexcept {
            if(!g->empty) {
                *output = g->bounds;
            }
        }

    private:
        Array<Interval<F>,3>* output;
    };

    template<typename F> using Reducer = CriticalReducer<F, 3, OnDone<F>>;

    template<typename F> void __global__ cuda_kernel_test_critical_reducer(const Array<Interval<F>, 3>* input, std::size_t in_length, Array<Interval<F>, 3>* output, typename Reducer<F>::GlobalMemory* global_buffer) {
        extern __shared__  char shared_buffer[];
        Reducer<F> reducer{shared_buffer, gridDim, output};
        reducer.initialize_shared();

        for(std::size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < in_length; i += blockDim.x * gridDim.x) {
            reducer.join(input[i]);
        }

        reducer.merge(global_buffer);
    }

    constexpr static std::size_t num_merged_crits = (65536 * 500);
    constexpr static std::size_t blocks = 256;
    constexpr static std::size_t threads_per_block = 256;
}

TEST_CASE_TEMPLATE("[ivarp][critical reducer] CUDA critical reducer", NT, float, double) {
    try {
        using Red = Reducer<NT>;
        std::size_t shared_space = Red::shared_memory_size_needed(threads_per_block);
        cuda::DeviceArray<typename Red::GlobalMemory> global_buffer(1);
        cuda::DeviceArray<Array<Interval<NT>,3>> output(1);
        cuda::DeviceArray<Array<Interval<NT>,3>> input(num_merged_crits);
        global_buffer[0] = GlobalMemoryInit{};

        Interval<NT> range{-100000.f, 100000.f};
        Interval<NT> widths[] = {
            {0, 0.001f}, {100.f, 10000.f}
        };
        NT m[3] = {impl::inf_value<NT>(), impl::inf_value<NT>(), impl::inf_value<NT>()};
        NT M[3] = {-impl::inf_value<NT>(), -impl::inf_value<NT>(), -impl::inf_value<NT>()};
        for(std::size_t i = 0; i < num_merged_crits; ++i) {
            for(std::size_t j = 0; j < 3; ++j) {
                Interval<NT> cur = random_interval_norestrict(range, std::begin(widths), std::end(widths));
                m[j] = (std::min)(cur.lb(), m[j]);
                M[j] = (std::max)(cur.ub(), M[j]);
                input[i][j] = cur;
            }
        }

        cudaDeviceSetSharedMemConfig(sizeof(NT) == 8 ? cudaSharedMemBankSizeEightByte : cudaSharedMemBankSizeFourByte);
        cuda_kernel_test_critical_reducer<<<blocks,threads_per_block,shared_space>>>(input.pass_to_device(), num_merged_crits, output.pass_to_device_nocopy(), global_buffer.pass_to_device());
        output.read_from_device();

        REQUIRE(output[0][0].lb() == m[0]);
        REQUIRE(output[0][0].ub() == M[0]);
        REQUIRE(output[0][1].lb() == m[1]);
        REQUIRE(output[0][1].ub() == M[1]);
        REQUIRE(output[0][2].lb() == m[2]);
        REQUIRE(output[0][2].ub() == M[2]);
    } catch(const CudaError& err) {
        std::cerr << "CudaError: " << err.what() << std::endl;
        REQUIRE(err.what() == nullptr);
    }
}
