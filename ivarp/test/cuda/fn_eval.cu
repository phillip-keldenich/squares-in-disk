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
#include "cuda_test_fn.cuh"

using namespace ivarp;

TEST_CASE_TEMPLATE("[ivarp][math_fn][cuda] Simple function eval", NT, float, double) {
    using IT = Interval<NT>;
    const auto simple_plus = args::x0 + args::x1 + args::x2 + 11_Z/19_Z;
    const auto simple_plusminus = -args::x0 + args::x1 - args::x2 + 111_Z/19_Z;
    const auto simple_mul = args::x0 * args::x1 * args::x2 * 315_Z/100_Z;
    const auto simple_div = args::x0 * args::x1 / args::x2 + 315_Z/100_Z;
    const auto minmax = minimum(args::x0, maximum(args::x0-1, args::x1, args::x2));

    Array<IT, 3> ranges{
        {{50.f, 75.f}, {-75.f, 0.f}, {-5.f, 5.f}}
    };

    IT widths[] = {
        {0.f, std::numeric_limits<NT>::min()},
        {0.f,0.1f}, {0.f, 3.f}
    };

    cuda_test_fn(simple_plus, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_plusminus, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_mul, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_div, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(minmax, 1u<<18u, ranges, std::begin(widths), std::end(widths));
}

TEST_CASE_TEMPLATE("[ivarp][math_fn][cuda] Simple predicate eval", NT, float, double) {
    using IT = Interval<NT>;

    const auto simple_leq = (args::x0 <= 33_Z/15_Z);
    const auto simple_eq = (args::x0 == -args::x0);
    const auto simple_not = !(args::x0 > 33_Z/15_Z);
    const auto simple_andorxor = ((args::x0 < 33_Z/15_Z || args::x0 > 57_Z) && square(args::x0) > 5_Z) ^ true;

    Array<IT, 1> ranges{{{-10.f,10.f}}};
    IT widths[] = {{0.f, std::numeric_limits<NT>::min()}, {1.f, 3.f}};

    cuda_test_fn(simple_leq, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_eq,  1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_not, 1u<<18u, ranges, std::begin(widths), std::end(widths));
    cuda_test_fn(simple_andorxor, 1u<<18u, ranges, std::begin(widths), std::end(widths));
}

using RootFn = MathSqrt<MathArgFromIndex<0>>;

void __global__ kernel_test_sqrt_evaluation(const RootFn root, int* output) {
    *output = -2;

    float ftest = 2.0f;
    IFloat iftest{2.0f,2.0f};
    IFloat iftestr{1.41421356f, 1.41421366f};
    double dtest = 2.0f;
    IDouble idtest{2.0,2.0};
    IDouble idtestr{1.4142135623730949, 1.4142135623730951};

    if(root.template array_evaluate<DefaultContextWithNumberType<float>>(&ftest) != 1.41421356f) {
        *output = 0;
        return;
    }
    if(root.template array_evaluate<DefaultContextWithNumberType<double>>(&dtest) != 1.4142135623730951) {
        *output = 1;
        return;
    }
    if(!root.template array_evaluate<DefaultContextWithNumberType<IFloat>>(&iftest).same(iftestr)) {
        *output = 2;
        return;
    }
    if(!root.template array_evaluate<DefaultContextWithNumberType<IDouble>>(&idtest).same(idtestr)) {
        *output = 3;
        return;
    }

    *output = -1;
}

TEST_CASE("[ivarp][math_fn][cuda] sqrt evaluation") {
    RootFn root = ivarp::sqrt(ivarp::args::x0);
    ivarp::cuda::DeviceArray<int> result(1);
    kernel_test_sqrt_evaluation<<<1,1>>>(root, result.pass_to_device_nocopy());
    throw_if_cuda_error("Error launching kernel (synchronous)", cudaPeekAtLastError());
    throw_if_cuda_error("Error launching kernel (asynchronous)", cudaDeviceSynchronize());
    result.read_from_device();
    REQUIRE(result[0] == -1);
}
