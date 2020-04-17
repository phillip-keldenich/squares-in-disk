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
// Created by Phillip Keldenich on 07.02.20.
//

#pragma once

#ifndef IVARP_CUDA_DEVICE_DISCOVERY_BEGIN_INCLUDE
#error "Do not directly include ivarp_cuda/cuda_device_discovery.hpp; include ivarp/cuda_device_discovery.hpp instead!"
#else

#include <cuda_runtime_api.h>
#include <exception>
#include <stdexcept>
#include <iostream>

namespace ivarp {
namespace impl {
    static inline IVARP_H bool device_is_viable(int id, const cudaDeviceProp& properties) noexcept {
        static constexpr int cuda_major_capability_required = 3;
        static constexpr int cuda_minor_capability_required = 5;

        return properties.major > cuda_major_capability_required ||
               (properties.major >= cuda_major_capability_required && properties.minor >= cuda_minor_capability_required);
    }
}

    static inline IVARP_H std::vector<int> viable_cuda_device_ids() noexcept {
        try {
            int num = -1;
            cudaError_t err = cudaGetDeviceCount(&num);
            if(err != cudaSuccess) {
                std::cerr << "WARNING: Could not get number of CUDA devices - error code " << err << ": " << cudaGetErrorString(err) << std::endl;
                return {};
            }
            std::vector<int> result;
            for(int id = 0; id < num; ++id) {
                cudaDeviceProp prop;
                if((err = cudaGetDeviceProperties(&prop, id)) != cudaSuccess) {
                    std::cerr << "WARNING: Could not query device properties for CUDA device " << id << " - error code " << err << ": "
                              << cudaGetErrorString(err) << std::endl;
                } else if(impl::device_is_viable(id, prop)) {
                    result.push_back(id);
                }
            }
            return result;
        } catch(std::exception& ex) {
            std::cerr << "WARNING: CUDA device detection failed: " << ex.what() << std::endl;
        } catch(...) {
            std::cerr << "WARNING: CUDA device detection failed with an unknown error!" << std::endl;
        }
        return {};
    }
}

#endif
