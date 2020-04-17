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
#include <ivarp/with_cuda/ivarp_cuda/memory.hpp>
#include "ivarp/number.hpp"
#include "ivarp_cuda/memory.hpp"
#include "ivarp_cuda/error.hpp"
#include "cuda_test_util.cuh"
#include <cstdint>

using namespace ivarp;

namespace {
    static constexpr int max_cuda_requires = 4096;

    template<typename FT> __global__ void kernel_exact_less_than_checks(FT /*typearg*/, int* results) {
        int cnt = 1;
        CUDA_REQUIRE(results, cnt, exact_less_than(FT(0), FT(1)));
        CUDA_REQUIRE(results, cnt, !exact_less_than(FT(1), FT(0)));
        CUDA_REQUIRE(results, cnt, exact_less_than(FT(0), 1));
        CUDA_REQUIRE(results, cnt, !exact_less_than(1, FT(0)));
        std::uint64_t too_large1 = std::uint64_t(1u) << 55u;
        std::uint64_t too_large2 = too_large1 + 1u;
        int okay_d = 87654321;
        double max_exact = 18014398509481984.0;
        double max_exactp = 18014398509481985.0;
        CUDA_REQUIRE(results, cnt, max_exact == max_exactp);
        double max_exactm1 = impl::step_down(max_exact);
        std::uint64_t okay_d64(okay_d);
        std::int64_t okay_i64(okay_d);
        CUDA_REQUIRE(results, cnt, (double)(float)okay_d != (double)okay_d);
        CUDA_REQUIRE(results, cnt,
                     exact_less_than(float(okay_d), okay_d) == ((double)(float)okay_d < (double)okay_d));
        CUDA_REQUIRE(results, cnt,
                     exact_less_than(float(okay_d), okay_d64) == ((double)(float)okay_d < (double)okay_d));
        CUDA_REQUIRE(results, cnt,
                     exact_less_than(float(okay_d), okay_i64) == ((double)(float)okay_d < (double)okay_d));
        CUDA_REQUIRE(results, cnt,
                     exact_less_than(-float(okay_d), -okay_i64) == ((double)(float)okay_d >= (double)okay_d));

        CUDA_REQUIRE(results, cnt, exact_less_than(FT(too_large1), too_large2));
        CUDA_REQUIRE(results, cnt, !exact_less_than(FT(too_large1), too_large1));
        CUDA_REQUIRE(results, cnt, !exact_less_than(FT(too_large2), too_large1));
        CUDA_REQUIRE(results, cnt, exact_less_than(FT(too_large2), too_large2));
        CUDA_REQUIRE(results, cnt, exact_less_than(FT(55), 750));
        CUDA_REQUIRE(results, cnt, exact_less_than(FT(-1333), 1333));
        CUDA_REQUIRE(results, cnt, !exact_less_than(FT(-1333), -1333));

        CUDA_REQUIRE(results, cnt, exact_less_than(-impl::max_value<FT>(), -INT64_MAX));
        CUDA_REQUIRE(results, cnt, exact_less_than(-impl::inf_value<FT>(), -INT64_MAX));
        CUDA_REQUIRE(results, cnt, exact_less_than(UINT64_MAX, impl::max_value<FT>()));
        CUDA_REQUIRE(results, cnt, exact_less_than(UINT64_MAX, impl::inf_value<FT>()));
        CUDA_REQUIRE(results, cnt, !exact_less_than(FT(UINT64_MAX/2+1), UINT64_MAX/2+1));
        CUDA_REQUIRE(results, cnt, exact_less_than(max_exactm1, max_exact));
        CUDA_REQUIRE(results, cnt, exact_less_than(-max_exact, -max_exactm1));
        CUDA_REQUIRE(results, cnt, exact_less_than(max_exact, too_large2));
        CUDA_REQUIRE(results, cnt, exact_less_than(max_exact, (1ull<<54u) + 1));
        CUDA_REQUIRE(results, cnt, exact_less_than(-(1ll<<54u) - 1, -max_exact)); // NOLINT
        CUDA_REQUIRE(results, cnt, !exact_less_than((1ll<<54u) + 1, max_exact)); // NOLINT
        CUDA_REQUIRE(results, cnt, exact_less_than(534.0, 1ull<<56u));
        CUDA_REQUIRE(results, cnt, exact_less_than(-1ll * (1ll<<56u), -999.9)); // NOLINT

        results[0] = cnt;
    }
}

TEST_CASE_TEMPLATE("[ivarp][cuda support] CUDA exact_less_than", FT, float, double) {
    cuda::DeviceArray<int> buffer(max_cuda_requires);
    kernel_exact_less_than_checks<<<1,1>>>(FT(0), buffer.pass_to_device_nocopy());
    buffer.read_from_device();

    REQUIRE(buffer[0] > 1);
    REQUIRE(buffer[0] < max_cuda_requires);

    for(int i = 1; i < buffer[0]; ++i) {
        REQUIRE(buffer[i] + i != i);
    }
}
