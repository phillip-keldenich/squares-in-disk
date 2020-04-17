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
// Created by Phillip Keldenich on 06.12.19.
//

#include <doctest/doctest_fixed.hpp>
#include <algorithm>
#include <ivarp/with_cuda/ivarp_cuda/memory.hpp>
#include "ivarp/number.hpp"
#include "ivarp/context.hpp"
#include "ivarp_cuda/memory.hpp"
#include "ivarp/math_fn.hpp"
#include "ivarp/tuple.hpp"
#include "../test_util.hpp"

using namespace ivarp;

namespace {
    static constexpr int num_values = 4099;
    static constexpr int num_blocks = 16;

    template<typename TT>
        void __global__ kernel_tuple_offsets(const TT* o, std::intptr_t* offsets)
    {
        offsets[0] = std::intptr_t(&get<0>(*o)) - std::intptr_t(o);
        offsets[1] = std::intptr_t(&get<1>(*o)) - std::intptr_t(o);
        offsets[2] = std::intptr_t(&get<2>(*o)) - std::intptr_t(o);
        offsets[3] = std::intptr_t(&get<3>(*o)) - std::intptr_t(o);
        offsets[4] = std::intptr_t(&get<4>(*o)) - std::intptr_t(o);
        offsets[5] = std::intptr_t(&get<5>(*o)) - std::intptr_t(o);
        offsets[6] = std::intptr_t(&get<6>(*o)) - std::intptr_t(o);
    }

    template<typename TT>
    void __global__ kernel_tuple_copy_htd(const TT* input, std::size_t s, int* result) {
        if(blockIdx.x == 0) {
            result[num_values] = (sizeof(TT) == s);
        }

        for(int i = blockIdx.x; i < num_values; i += num_blocks) {
            const TT& t = input[i];
            bool r = (get<0>(t) == 23 * i + 42);
            r &= get<1>(t).same(IFloat(i,i+1));
            r &= get<2>(t) == (0.5 * i);
            r &= get<3>(t) == (22.f * i);
            r &= get<4>(t).same(IDouble(-i-1, i+1));
            r &= get<5>(t) == 'c';
            r &= !get<6>(t);
            result[i] = r;
        }
    }

    template<typename TT>
    void __global__ kernel_tuple_copy_dth(TT* output) {
        for(int i = blockIdx.x; i < num_values; i += num_blocks) {
            TT& t = output[i];
            get<0>(t) = IFloat{-i+1, i+22};
            get<1>(t) = (i % 2 == 0);
            get<2>(t) = IDouble{-0.25*i, 0.75*i};
        }
    }
}

TEST_CASE("[ivarp][cuda support] Tuple offsets host vs. device") {
	using TT = Tuple<int, IFloat, double, float, IDouble, char, bool>;
	try {
        cuda::DeviceArray<TT> in_offsets(1);
        cuda::DeviceArray<std::intptr_t> out_offsets(7);
        kernel_tuple_offsets<<<1,1>>>(in_offsets.pass_to_device_nocopy(), out_offsets.pass_to_device_nocopy());
        out_offsets.read_from_device();
        TT oo, *o = &oo;
        std::intptr_t exp_offsets[7] = {
            std::intptr_t(&get<0>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<1>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<2>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<3>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<4>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<5>(*o)) - std::intptr_t(o),
            std::intptr_t(&get<6>(*o)) - std::intptr_t(o)
        };

		for(int i = 0; i < 7; ++i) {
			REQUIRE(out_offsets[i] == exp_offsets[i]);
		}
	} catch(const CudaError& error) {
        std::cerr << "Exception: CUDA error!" << std::endl << error.what() << std::endl;
        throw error;
    }
}

TEST_CASE("[ivarp][cuda support] Tuple copy host to device") {
    using TT = Tuple<int, IFloat, double, float, IDouble, char, bool>;

    try {
        cuda::DeviceArray<TT> in(num_values);
        cuda::DeviceArray<int> out(num_values+1);

        std::fill(out.begin(), out.end(), 0);
        for(int i = 0; i < num_values; ++i) {
            TT& t = in[i];
            get<0>(t) = 23 * i + 42;
            get<1>(t) = IFloat{i,i+1};
            get<2>(t) = 0.5 * i;
            get<3>(t) = 22.f * i;
            get<4>(t) = IDouble{-i-1,i+1};
            get<5>(t) = 'c';
            get<6>(t) = false;
        }
        dim3 blocks = num_blocks;
        kernel_tuple_copy_htd<<<blocks,1>>>(in.pass_to_device(), sizeof(TT), out.pass_to_device_nocopy());
        out.read_from_device();

        for(int i = 0; i < num_values+1; ++i) {
            REQUIRE(out[i] == 1);
        }
    } catch(const CudaError& error) {
        std::cerr << "Exception: CUDA error!" << std::endl << error.what() << std::endl;
        throw error;
    }
}

TEST_CASE("[ivarp][cuda support] Tuple copy device to host") {
    using TT = Tuple<IFloat, bool, IDouble>;

    try {
        cuda::DeviceArray<TT> values(num_values);
        dim3 blocks = num_blocks;
        kernel_tuple_copy_dth<<<blocks,1>>>(values.pass_to_device_nocopy());
        values.read_from_device();

        for(int i = 0; i < num_values; ++i) {
            const TT& t = values[i];
            REQUIRE_SAME(get<0>(t), IFloat(-i+1, i+22));
            REQUIRE(get<1>(t) == (i % 2 == 0));
            REQUIRE_SAME(get<2>(t), IDouble(-0.25*i, 0.75*i));
        }
    } catch(const CudaError& error) {
        std::cerr << "Exception: CUDA error!" << std::endl << error.what() << std::endl;
        throw error;
    }
}
