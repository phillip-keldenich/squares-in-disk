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
// Created by Phillip Keldenich on 14.12.19.
//

#include "test_util.hpp"
#include "ivarp/cuda_transformer.hpp"

using namespace ivarp;

template<typename NumberType> static inline void check_output(const ivarp::IRational& r, ivarp::Interval<NumberType> gpu) {
    REQUIRE((!r.possibly_undefined() || gpu.possibly_undefined()));
    REQUIRE(gpu.lb() <= gpu.ub());

    REQUIRE((r.finite_lb() || !gpu.finite_lb()));
    REQUIRE((r.finite_ub() || !gpu.finite_ub()));
    REQUIRE(!gpu.below_lb(r.lb()));
    REQUIRE(!gpu.above_ub(r.ub()));
}

static inline void check_output(ivarp::IBool r, ivarp::IBool gpu) {
    if(!ivarp::definitely(r)) {
        REQUIRE(!ivarp::definitely(gpu));
    }
    if(ivarp::possibly(r)) {
        REQUIRE(ivarp::possibly(gpu));
    }
}

template<typename MathExprOrPred, std::size_t NumVars, typename NumberType, typename WidthsIter> static inline void IVARP_H
    cuda_transform_test_fn(const MathExprOrPred& expr, std::size_t num_samples, Array<Interval<NumberType>, NumVars> ranges,
                           WidthsIter widths_begin, WidthsIter widths_end)
{
    using NumberCtx = DefaultContextWithNumberType<Interval<NumberType>>;
    using RationalCtx = DefaultContextWithNumberType<IRational>;
    const auto transformed_fn = transform_for_cuda(expr);
    static_assert(std::decay_t<decltype(transformed_fn)>::cuda_supported, "After transformation, CUDA should be supported!");

    for(std::size_t i = 0; i < num_samples; ++i) {
        Array<Interval<NumberType>, NumVars> in;
        Array<IRational, NumVars> rin;
        for(std::size_t j = 0; j < NumVars; ++j) {
            in[j] = random_interval(ranges[j], widths_begin, widths_end);
            rin[j] = convert_number<IRational>(in[j]);
        }
        auto tres = transformed_fn.template array_evaluate<NumberCtx>(in);
        auto rres = expr.template array_evaluate<RationalCtx>(rin);
        check_output(rres, tres);
    }
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] simple function eval", NT, float, double) {
    using IT = Interval<NT>;
    const auto simple_plus = args::x0 + args::x1 + args::x2 + 11_Z/19_Z;
    const auto simple_plusminus = -args::x0 + args::x1 - args::x2 + 111_Z/19_Z;
    const auto simple_mul = args::x0 * args::x1 * args::x2 * 315_Z/100_Z;
    const auto simple_div = args::x0 * args::x1 / args::x2 + 3.15_X;
    const auto minmax = minimum(args::x0, maximum(args::x0-1, args::x1, args::x2));

    Array<IT, 3> ranges{
        {{50.f, 75.f}, {-75.f, 0.f}, {-5.f, 5.f}}
    };

    IT widths[] = {
        {0.f, std::numeric_limits<NT>::min()},
        {0.f,0.1f}, {0.f, 3.f}
    };

    cuda_transform_test_fn(simple_plus, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_plusminus, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_mul, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_div, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(minmax, 1u<<14u, ranges, std::begin(widths), std::end(widths));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn] simple predicate eval", NT, float, double) {
    using IT = Interval<NT>;

    const auto simple_leq = (args::x0 <= 33_Z/15_Z);
    const auto simple_eq = (args::x0 == -args::x0);
    const auto simple_not = !(args::x0 > 33_Z/15_Z);
    const auto simple_andorxor = ((args::x0 < 33_Z/15_Z || args::x0 > 57_Z) && square(args::x0) > 5_Z) ^ true;

    using T = std::decay_t<decltype(simple_andorxor)>;
    using ExpectedT = ivarp::BinaryMathPred<
        ivarp::MathPredXor,
        ivarp::NAryMathPred<
            ivarp::MathPredAndSeq,
            ivarp::NAryMathPred<
                ivarp::MathPredOrSeq,
                ivarp::BinaryMathPred<
                    ivarp::BinaryMathPredLT,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >,
                    ivarp::MathConstant<Rational,2'200000,2'200000>
                >,
                ivarp::BinaryMathPred<
                    ivarp::BinaryMathPredGT,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >,
                    ivarp::MathConstant<Rational, 57'000000, 57'000000>
                >
            >,
            ivarp::BinaryMathPred<
                ivarp::BinaryMathPredGT,
                ivarp::MathUnary<
                    ivarp::MathFixedPowTag<2>,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >
                >,
                ivarp::MathConstant<Rational, 5'000000, 5'000000>
            >
        >,
        ivarp::MathBoolConstant<bool, false, true>
    >;
    using TransformT = std::decay_t<decltype(transform_for_cuda(simple_andorxor))>;
    using ExpectedTransformT = ivarp::BinaryMathPred<
        ivarp::MathPredXor,
        ivarp::NAryMathPred<
            ivarp::MathPredAndSeq,
            ivarp::NAryMathPred<
                ivarp::MathPredOrSeq,
                ivarp::BinaryMathPred<
                    ivarp::BinaryMathPredLT,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >,
                    ivarp::MathCUDAConstant<2'200000, 2'200000>
                >,
                ivarp::BinaryMathPred<
                    ivarp::BinaryMathPredGT,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >,
                    ivarp::MathCUDAConstant<57'000000, 57'000000>
                >
            >,
            ivarp::BinaryMathPred<
                ivarp::BinaryMathPredGT,
                ivarp::MathUnary<
                    ivarp::MathFixedPowTag<2>,
                    ivarp::MathArg<std::integral_constant<unsigned int, 0> >
                >,
                ivarp::MathCUDAConstant<5'000000, 5'000000>
            >
        >,
        ivarp::MathBoolConstant<bool, false, true>
    >;
    static_assert(std::is_same<T,ExpectedT>::value, "Wrong type!");
    static_assert(std::is_same<TransformT, ExpectedTransformT>::value, "Wrong transformed type!");

    Array<IT, 1> ranges{{{-10.f,10.f}}};
    IT widths[] = {{0.f, std::numeric_limits<NT>::min()}, {1.f, 3.f}};

    cuda_transform_test_fn(simple_leq, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_eq,  1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_not, 1u<<14u, ranges, std::begin(widths), std::end(widths));
    cuda_transform_test_fn(simple_andorxor, 1u<<14u, ranges, std::begin(widths), std::end(widths));
}
