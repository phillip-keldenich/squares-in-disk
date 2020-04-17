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
// Created by Phillip Keldenich on 28.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
namespace cuda {
    template<typename StaticSplitInfos> static inline constexpr IVARP_H int get_grid_y(StaticSplitInfos) {
        return 1;
    }

    template<typename S1, typename S2, typename S3, typename S4, typename... S> static inline constexpr IVARP_H int
        get_grid_y(SplitInfoSequence<S1,S2,S3,S4,S...>)
    {
        return S1::subvisisions > 4 ? 4 : S1::subdivisions;
    }

    template<std::size_t StaticSplitIndex, typename StaticSplitInfo> struct GridInfo {
        static constexpr IVARP_D int initial() noexcept { return 0; }
        static constexpr IVARP_D int increment() noexcept { return 1; }
    };

    template<typename S1> struct GridInfo<0, SplitInfoSequence<S1>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.x; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.x; }
    };

    template<typename S1, typename S2> struct GridInfo<0, SplitInfoSequence<S1,S2>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.y; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.y; }
    };

    template<typename S1, typename S2, typename S3> struct GridInfo<0, SplitInfoSequence<S1,S2,S3>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.z; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.z; }
    };

    template<typename S1, typename S2, typename S3, typename S4, typename... S>
    struct GridInfo<0, SplitInfoSequence<S1,S2,S3,S4,S...>> {
        static constexpr IVARP_D int initial() noexcept { return blockIdx.y; }
        static constexpr IVARP_D int increment() noexcept { return gridDim.y; }
    };

    template<typename S1, typename S2> struct GridInfo<1, SplitInfoSequence<S1,S2>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.x; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.x; }
    };

    template<typename S1, typename S2, typename S3> struct GridInfo<1, SplitInfoSequence<S1,S2,S3>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.y; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.y; }
    };

    template<typename S1, typename S2, typename S3, typename S4, typename... S>
    struct GridInfo<1, SplitInfoSequence<S1,S2,S3,S4,S...>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.z; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.z; }
    };

    template<typename S1, typename S2, typename S3> struct GridInfo<2, SplitInfoSequence<S1,S2,S3>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.x; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.x; }
    };

    template<typename S1, typename S2, typename S3, typename S4, typename... S>
    struct GridInfo<2, SplitInfoSequence<S1,S2,S3,S4,S...>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.y; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.y; }
    };

    template<typename S1, typename S2, typename S3, typename S4, typename... S>
    struct GridInfo<3, SplitInfoSequence<S1,S2,S3,S4,S...>> {
        static constexpr IVARP_D int initial() noexcept { return threadIdx.x; }
        static constexpr IVARP_D int increment() noexcept { return blockDim.x; }
    };

    template<typename CurrentSplit, typename SplitInfoSeq, std::size_t Index=0> struct FindSplitIndex;
    template<typename CurrentSplit, typename S1, typename... S, std::size_t Index>
    struct FindSplitIndex<CurrentSplit, SplitInfoSequence<S1,S...>, Index> {
    private:
        struct IsCurrent {
            template<typename = void> struct Lazy {
                static constexpr std::size_t value = Index;
            };
        };
        struct IsNotCurrent {
            template<typename = void> struct Lazy {
                static constexpr std::size_t value =
                    FindSplitIndex<CurrentSplit, SplitInfoSequence<S...>, Index+1>::value;
            };
        };
        using Lazy = std::conditional_t<std::is_same<S1,CurrentSplit>::value, IsCurrent, IsNotCurrent>;

    public:
        static constexpr std::size_t value = Lazy::template Lazy<>::value;
    };

    template<typename CurrentSplit, typename StaticSplitInfo> using GridInfoOf =
        GridInfo<FindSplitIndex<CurrentSplit, StaticSplitInfo>::value, StaticSplitInfo>;

    static inline cudaDeviceProp get_properties(int device_id) {
        cudaDeviceProp result;
        throw_if_cuda_error("Could not query device properties", cudaGetDeviceProperties(&result, device_id));
        return result;
    }

    static inline unsigned fix_threads_per_block(const cudaDeviceProp& p, int threads_per_block) {
        if(threads_per_block < p.warpSize) {
            threads_per_block = p.warpSize;
        }
        std::int64_t max_per_dim = p.maxThreadsDim[0];
        max_per_dim *= p.maxThreadsDim[1];
        max_per_dim *= p.maxThreadsDim[2];
        int max_threads_per_block = p.maxThreadsPerBlock;
        if(std::int64_t(max_threads_per_block) > max_per_dim) {
            max_threads_per_block = static_cast<int>(max_per_dim);
        }

        if(max_threads_per_block < threads_per_block) {
            threads_per_block = max_threads_per_block;
        }
        if(threads_per_block % p.warpSize != 0) {
            int num_warps = (threads_per_block / p.warpSize) + 1;
            threads_per_block = num_warps * p.warpSize;
        }
        return static_cast<unsigned>(threads_per_block);
    }

    template<typename SI> static inline unsigned actual_subdivisions(const cudaDeviceProp& p, int dim_index) noexcept {
        auto config_s = static_cast<unsigned>(SI::subdivisions);
        auto max_s = static_cast<unsigned>(p.maxThreadsDim[dim_index]);
        return std::min(config_s, max_s);
    }

    static inline dim3 pack_factors_2(unsigned threads_per_block, unsigned split1, unsigned split2) {
        unsigned bucket_in[2] = {split1, split2};
        unsigned bucket_out[2];
        pack_factors(threads_per_block, 2, bucket_in, bucket_out);
        return dim3(int(bucket_out[1]), int(bucket_out[0]));
    }

    static inline dim3 pack_factors_3(unsigned threads_per_block, unsigned split1, unsigned split2, unsigned split3) {
        unsigned bucket_in[3] = {split1, split2, split3};
        unsigned bucket_out[3];
        pack_factors(threads_per_block, 3, bucket_in, bucket_out);
        return dim3(int(bucket_out[2]), int(bucket_out[1]), int(bucket_out[0]));
    }

    template<typename S1> static inline constexpr IVARP_H dim3
        get_block_dims(int device_id, int threads_per_block_setting, SplitInfoSequence<S1>)
    {
        auto p = get_properties(device_id);
        unsigned threads_per_block = fix_threads_per_block(p, threads_per_block_setting);
        return dim3(int(std::min(threads_per_block, actual_subdivisions<S1>(p, 0))));
    }

    template<typename S1, typename S2> static inline constexpr IVARP_H dim3
        get_block_dims(int device_id, int threads_per_block_setting, SplitInfoSequence<S1,S2>)
    {
        auto p = get_properties(device_id);
        unsigned threads_per_block = fix_threads_per_block(p, threads_per_block_setting);
        unsigned ts1 = actual_subdivisions<S1>(p, 1);
        unsigned ts2 = actual_subdivisions<S2>(p, 0);

        std::uint64_t subs = std::uint64_t{ts1} * ts2;
        if(subs <= threads_per_block) {
            return dim3(int(ts2), int(ts1));
        } else {
            return pack_factors_2(threads_per_block, ts1, ts2);
        }
    }

    template<typename S1, typename S2, typename S3> static inline constexpr IVARP_H dim3
        get_block_dims(int device_id, int threads_per_block_setting, SplitInfoSequence<S1,S2,S3>)
    {
        auto p = get_properties(device_id);
        unsigned threads_per_block = fix_threads_per_block(p, threads_per_block_setting);
        unsigned ts1 = actual_subdivisions<S1>(p, 2);
        unsigned ts2 = actual_subdivisions<S2>(p, 1);
        unsigned ts3 = actual_subdivisions<S3>(p, 0);
        std::uint64_t subs = std::uint64_t{ts1} * ts2 * ts3;
        if(subs <= threads_per_block) {
            return dim3(int(ts3), int(ts2), int(ts1));
        } else {
            return pack_factors_3(threads_per_block, ts1, ts2, ts3);
        }
    }

    template<typename S1, typename S2, typename S3, typename S4, typename... S> static inline constexpr IVARP_H dim3
        get_block_dims(int device_id, int threads_per_block_setting, SplitInfoSequence<S1,S2,S3,S4,S...>)
    {
        return get_block_dims(device_id,  threads_per_block_setting, SplitInfoSequence<S2,S3,S4>{});
    }
}
}
}

