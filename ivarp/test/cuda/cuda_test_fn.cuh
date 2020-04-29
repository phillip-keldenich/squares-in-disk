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
#include "ivarp_cuda/memory.hpp"
#include "ivarp/math_fn.hpp"
#include "ivarp/cuda_transformer.hpp"

namespace {
    template<typename Fn, typename NT, typename R, std::size_t NumVars> void __global__ kernel_cuda_test_fn(Fn fn, std::size_t num_samples,
                                                                                                            const ivarp::Array<ivarp::Interval<NT>, NumVars>* in,
                                                                                                            R* out)
    {
        using Context = ivarp::DefaultContextWithNumberType<ivarp::Interval<NT>>;
        static_assert(std::decay_t<Fn>::cuda_supported, "After transformation, CUDA should be supported!");

        for(std::size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < num_samples; i += blockDim.x * gridDim.x) {
            out[i] = fn.template array_evaluate<Context>(in[i]);
        }
    }
}

static inline bool check_output(ivarp::IBool r, ivarp::IBool gpu) {
    if(!ivarp::definitely(r)) {
        REQUIRE(!ivarp::definitely(gpu));
    }
    if(ivarp::possibly(r)) {
        REQUIRE(ivarp::possibly(gpu));
    }
	return true;
}

template<typename NumberType> static inline bool check_output(const ivarp::IRational& r, ivarp::Interval<NumberType> gpu) {
	if(
		(r.possibly_undefined() && !gpu.possibly_undefined()) ||
		(gpu.lb() > gpu.ub()) ||
		(!r.finite_lb() && gpu.finite_lb()) ||
		(!r.finite_ub() && gpu.finite_ub()) ||
		gpu.below_lb(r.lb()) ||
		gpu.above_ub(r.ub())
	) {
		std::cerr << "CUDA function returned invalid or wrong result - CUDA " << gpu << ", rational (CPU) " << r << std::endl;
		return false;
	}
	return true;
}

template<typename NumberType, typename MathExpr, typename WidthsIter, std::size_t NumVars>
    static inline IVARP_H void cuda_test_fn(const MathExpr& expr, std::size_t num_samples, ivarp::Array<ivarp::Interval<NumberType>, NumVars> ranges,
                                            WidthsIter widths_begin, WidthsIter widths_end)
{
    const auto cuda_expr = ivarp::transform_for_cuda(expr);
    static_assert(std::decay_t<decltype(cuda_expr)>::cuda_supported, "After transformation, CUDA should be supported!");

    using ResultType = std::conditional_t<ivarp::IsMathPred<decltype(cuda_expr)>::value, ivarp::IBool, ivarp::Interval<NumberType>>;
    using RationalCtx = ivarp::DefaultContextWithNumberType<ivarp::IRational>;

    ivarp::cuda::DeviceArray<ivarp::Array<ivarp::Interval<NumberType>, NumVars>> input(num_samples);
    ivarp::cuda::DeviceArray<ResultType> output(num_samples);

    for(std::size_t i = 0; i < num_samples; ++i) {
        for(std::size_t j = 0; j < NumVars; ++j) {
            input[i][j] = ivarp::random_interval(ranges[j], widths_begin, widths_end);
        }
    }

    kernel_cuda_test_fn<<<128,128>>>(cuda_expr, num_samples, input.pass_to_device(), output.pass_to_device_nocopy());
    output.read_from_device();

    for(std::size_t i = 0; i < num_samples; ++i) {
        ivarp::Array<ivarp::IRational, NumVars> ival;
        for(std::size_t j = 0; j < NumVars; ++j) {
            ival[j] = ivarp::convert_number<ivarp::IRational>(input[i][j]);
        }

        auto result = expr.template array_evaluate<RationalCtx>(ival);
        if(!check_output(result, output[i])) {
			REQUIRE(false);
		}
    }
}

