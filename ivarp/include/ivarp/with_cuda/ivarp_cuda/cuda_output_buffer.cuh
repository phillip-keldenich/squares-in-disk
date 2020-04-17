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
// Created by Phillip Keldenich on 27.02.20.
//

#pragma once

#include "ivarp/cuda.hpp"
#include "ivarp_cuda/memory.hpp"
#include "ivarp_cuda/error.hpp"
#include <cstddef>
#include <algorithm>
#include <boost/range/iterator_range.hpp>

namespace ivarp {
namespace cuda {
    template<typename T> class CUDAOutputBuffer {
    public:
        explicit CUDAOutputBuffer(std::size_t init_size, std::size_t limit) :
            storage(init_size),
            total_limit(limit),
            host_side(*this),
            device_side(alloc_device_ptr())
        {}

        ~CUDAOutputBuffer() {
            cudaFree(device_side);
        }

        CUDAOutputBuffer(CUDAOutputBuffer&&) = delete;
        CUDAOutputBuffer &operator=(CUDAOutputBuffer&&) = delete;

        class CUDADeviceOutputBuffer {
        public:
            CUDADeviceOutputBuffer() = default;
            CUDADeviceOutputBuffer(const CUDADeviceOutputBuffer&) noexcept = default;
            CUDADeviceOutputBuffer &operator=(const CUDADeviceOutputBuffer&) noexcept = default;
            ~CUDADeviceOutputBuffer() = default;

            explicit CUDADeviceOutputBuffer(const CUDAOutputBuffer& from) noexcept :
                buffer(from.storage.pass_to_device_nocopy()),
                current(0),
                begin_at(0),
                buffer_size(from.storage.size())
            {}

            IVARP_D void push_back(const T& t) noexcept {
                std::size_t index = gpu_atomic_add(&current, std::size_t(1));
                if(index >= begin_at) {
                    std::size_t i = index - begin_at;
                    if(i < buffer_size) {
                        buffer[i] = t;
                    }
                }
            }

        private:
            friend CUDAOutputBuffer;
            T* buffer;
            std::size_t current, begin_at, buffer_size;
        };

        CUDADeviceOutputBuffer* pass_to_device() {
            host_side.current = 0;
            throw_if_cuda_error("Could not copy output buffer information to device (or asynchronous error)",
                cudaMemcpy(device_side, static_cast<const void*>(&host_side),
                           sizeof(CUDADeviceOutputBuffer), cudaMemcpyHostToDevice)
            );
            return device_side;
        }

        bool buffer_was_sufficient() {
            return host_side.current <= host_side.begin_at + host_side.buffer_size;
        }

        void prepare_next_call() {
            std::size_t elements_done = host_side.begin_at + host_side.buffer_size;
            std::size_t elements_total = host_side.current;
            std::size_t needed = elements_total - elements_done;
            if(needed > storage.size()) {
                needed = (std::min)(needed, total_limit);
                storage.grow_drop(needed);
                host_side.buffer = storage.pass_to_device_nocopy();
            }
            host_side.begin_at = elements_done;
        }

        void reset() {
            host_side.begin_at = 0;
        }

        auto get_output() {
            throw_if_cuda_error("Could not copy output buffer information to host (or asynchronous error)",
                cudaMemcpy(&host_side, static_cast<const void*>(device_side),
                           sizeof(CUDADeviceOutputBuffer), cudaMemcpyDeviceToHost)
            );

            if(host_side.current == 0) {
                return boost::make_iterator_range(storage.begin(), storage.begin());
            }
            storage.read_from_device();
            std::size_t content_length = host_side.current - host_side.begin_at;
            return boost::make_iterator_range(storage.begin(), storage.begin() + content_length);
        }

    private:
        static CUDADeviceOutputBuffer* alloc_device_ptr() {
            void* target;
            throw_if_cuda_error("Could not allocate device memory for output buffer",
					            cudaMalloc(&target, sizeof(CUDADeviceOutputBuffer)));
            return static_cast<CUDADeviceOutputBuffer*>(target);
        }

        DeviceArray<T> storage;
        std::size_t total_limit;
        CUDADeviceOutputBuffer host_side;
        CUDADeviceOutputBuffer *device_side;
    };
}
}

