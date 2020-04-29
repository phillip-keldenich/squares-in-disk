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
#include "ivarp/number.hpp"
#include "ivarp/context.hpp"
#include "ivarp_cuda/memory.hpp"
#include "ivarp/math_fn.hpp"
#include "../test_util.hpp"

using namespace ivarp;

namespace {
    struct OpAdd {
        template<typename NT> IVARP_HD NT operator()(NT n1, NT n2) const {
            return n1 + n2;
        }
    };

    struct OpSub {
        template<typename NT> IVARP_HD NT operator()(NT n1, NT n2) const {
            return n1 - n2;
        }
    };

	struct OpMul {
		template<typename NT> IVARP_HD NT operator()(NT n1, NT n2) const {
			return n1 * n2;
		}
	};

	struct OpDiv {
	    template<typename NT> IVARP_HD NT operator()(NT n1, NT n2) const {
	        return n1 / n2;
	    }
	};

	struct OpSqrt {
	    template<typename NT> IVARP_HD NT operator()(NT n1) const {
	        return ivarp::sqrt<DefaultContextWithNumberType<NT>>(n1);
	    }
	};

    template<typename Op, typename FloatType>
        void __global__ simple_interval_test(const Op op, const Interval<FloatType>* inputs, Interval<FloatType>* output,
                                             std::size_t in_length)
    {
        for(std::size_t i = blockIdx.x; i < in_length; i += gridDim.x) {
            std::size_t base_offset = i * in_length;
            for(std::size_t j = threadIdx.x; j < in_length; j += blockDim.x) {
                output[base_offset + j] = op(inputs[i], inputs[j]);
            }
        }
    }

    template<typename Op,typename FloatType>
        void __global__ unary_interval_test(const Op op, const Interval<FloatType>* inputs, Interval<FloatType>* output, std::size_t io_length)
    {
        const std::size_t per_block = io_length / gridDim.x;
        std::size_t i = per_block * blockIdx.x;
        std::size_t e = i + per_block;
        if(e > io_length) {
            e = io_length;
        }

        for(i += threadIdx.x; i < e; i += blockDim.x) {
            output[i] = op(inputs[i]);
        }
    }

    template<typename Op, typename FloatType>
        void run_simple_interval_test(const cuda::DeviceArray<Interval<FloatType>>& input,
                                      cuda::DeviceArray<Interval<FloatType>>& output)
    {
        const std::size_t n = input.size();
        const Op op{};

        const dim3 grid_dims = 64;
        const dim3 block_dims = 64;

        simple_interval_test<<<grid_dims,block_dims>>>(Op{}, input.pass_to_device(), output.pass_to_device_nocopy(), n);
        output.read_from_device();

        for(std::size_t i = 0, offs = 0; i < n; ++i) {
            for(std::size_t j = 0; j < n; ++j, ++offs) {
                REQUIRE_SAME(output[offs],  op(input[i], input[j]));
            }
        }
    }

    template<typename Op, typename FloatType>
        void run_unary_interval_test(const cuda::DeviceArray<Interval<FloatType>>& input, cuda::DeviceArray<Interval<FloatType>>& output)
    {
        const std::size_t n = input.size();
        const Op op{};

        const dim3 grid_dims = 64;
        const dim3 block_dims = 64;

        unary_interval_test<<<grid_dims,block_dims>>>(Op{}, input.pass_to_device(), output.pass_to_device_nocopy(), n);
        output.read_from_device();

        for(std::size_t i = 0; i < n; ++i) {
            REQUIRE_SAME(output[i], op(input[i]));
        }
    }
}

TEST_CASE_TEMPLATE("[ivarp][number] CUDA test (arithmetic operations), device vs. host", NT, float, double) {
    using IT = Interval<NT>;
    try {
        IT widths[] = {
                {0.000001f, 0.00001f},
                {0.01f, 0.1f},
                {0.1f, 0.8f},
                {1.f, 2.f},
                {10.f, 50.f}
        };
        IT range{-100.f,100.f};
        constexpr std::size_t num_intervals = 1u << 12u;

        cuda::DeviceArray<IT> intervals(num_intervals);
        cuda::DeviceArray<IT> result(num_intervals * num_intervals);
        for(std::size_t i = 0; i < num_intervals; ++i) {
            intervals[i] = random_interval(range, std::begin(widths), std::end(widths));
        }

        run_simple_interval_test<OpAdd>(intervals, result);
        run_simple_interval_test<OpSub>(intervals, result);
		run_simple_interval_test<OpMul>(intervals, result);
		run_simple_interval_test<OpDiv>(intervals, result);
    } catch(const CUDAError& error) {
        std::cerr << "Exception: CUDA error!" << std::endl << error.what() << std::endl;
        throw error;
    }
}

TEST_CASE_TEMPLATE("[ivarp][number] CUDA test (sqrt), device vs. host", NT, float, double) {
    using IT = Interval<NT>;
    try {
        IT widths[] = {
                {0.000001f, 0.00001f},
                {0.01f, 0.1f},
                {0.1f, 0.8f},
                {1.f, 2.f},
                {10.f, 50.f},
                {100.f,500.f}
        };
        IT range{0.f,1000.f};
        constexpr std::size_t num_intervals = 1u << 20u;

        cuda::DeviceArray<IT> intervals(num_intervals);
        cuda::DeviceArray<IT> result(num_intervals);
        for(std::size_t i = 0; i < num_intervals; ++i) {
            intervals[i] = random_interval(range, std::begin(widths), std::end(widths));
        }

        run_unary_interval_test<OpSqrt>(intervals, result);
    } catch(const CUDAError& error) {
        std::cerr << "Exception: CUDA error!" << std::endl << error.what() << std::endl;
        throw error;
    }
}
