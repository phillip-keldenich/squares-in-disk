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
// Created by Phillip Keldenich on 25.11.19.
//

#pragma once

#include "ivarp/cuda.hpp"
#include "ivarp_cuda/error.hpp"
#include <memory>
#include <type_traits>

namespace ivarp {
namespace cuda {
	struct CUDAFreeDeleter {
		template<typename T> void operator()(const T* ptr) noexcept {
		    if(ptr) {
                cudaFree(static_cast<void *>(const_cast<T *>(ptr)));
            }
		}
	};

	/**
	 * An array of objects in dynamically-allocated memory that exists on both host and device side.
	 * @tparam T
	 */
    template<typename T> class DeviceArray {
        static_assert(std::is_trivially_copy_constructible<T>::value,
                      "Non-trivial types cannot be copied between host and device!");
        static_assert(std::is_trivially_copy_assignable<T>::value,
                      "Non-trivial types cannot be copied between host and device!");
        static_assert(std::is_trivially_destructible<T>::value,
                      "Non-trivial types cannot be copied between host and device!");

    public:
        IVARP_H DeviceArray() noexcept :
            length(0),
            mem_on_host(0),
            mem_on_device(0)
        {}

        IVARP_H explicit DeviceArray(std::size_t length) :
            length(length),
            mem_on_host(std::make_unique<T[]>(length)),
            mem_on_device(alloc_device_ptr(length))
        {}

        template<typename RAIter>
        IVARP_H DeviceArray(RAIter b, RAIter e) :
            length(static_cast<std::size_t>(e-b)),
            mem_on_host(std::make_unique<T[]>(length)),
            mem_on_device(alloc_device_ptr(length))
        {
            std::copy(b, e, mem_on_host);
        }

        DeviceArray(const DeviceArray&) = delete;
        DeviceArray &operator=(const DeviceArray&) = delete;
        DeviceArray(DeviceArray&& m) noexcept = default;
        DeviceArray &operator=(DeviceArray&& m) noexcept = default;
        ~DeviceArray() = default;

        IVARP_H T& operator[](std::size_t s) noexcept {
            return mem_on_host[s];
        }

        IVARP_H const T& operator[](std::size_t s) const noexcept {
            return mem_on_host[s];
        }

        IVARP_H void read_from_device() {
            read_n_from_device(length);
        }

        IVARP_H void read_n_from_device(std::size_t n) {
            throw_if_cuda_error("Could not copy memory from device to host",
                cudaMemcpy(mem_on_host.get(), mem_on_device.get(), n * sizeof(T), cudaMemcpyDeviceToHost)
            );
        }

        IVARP_H T* pass_to_device() const {
            return pass_to_device_copy_n(length);
        }

        IVARP_H T* pass_to_device_copy_n(std::size_t n) const {
            throw_if_cuda_error("Could not copy memory from host to device",
                                cudaMemcpy(mem_on_device.get(), mem_on_host.get(), n * sizeof(T), cudaMemcpyHostToDevice));
            return mem_on_device.get();
        }

        IVARP_H T* pass_to_device_nocopy() const noexcept {
            return mem_on_device.get();
        }

        IVARP_H std::size_t size() const noexcept {
            return length;
        }

        IVARP_H T* begin() noexcept {
            return mem_on_host.get();
        }

        IVARP_H T* end() noexcept {
            return mem_on_host.get() + length;
        }

        IVARP_H const T* begin() const noexcept {
            return mem_on_host.get();
        }

        IVARP_H const T* end() const noexcept {
            return mem_on_host.get() + length;
        }

        IVARP_H const T* cbegin() const noexcept {
            return mem_on_host.get();
        }

        IVARP_H const T* cend() const noexcept {
            return mem_on_host.get() + length;
        }

        // Grow the buffer, dropping content on host and device
        IVARP_H void grow_drop(std::size_t new_length) {
            if(new_length <= length) {
                return;
            }

            auto new_host_mem = std::make_unique<T[]>(new_length);
            std::unique_ptr<T[], CUDAFreeDeleter> new_dev_mem(alloc_device_ptr(new_length));
            length = new_length;
            mem_on_host = std::move(new_host_mem);
            mem_on_device = std::move(new_dev_mem);
        }

        // Grow the buffer, keeping content on host and device
        IVARP_H void grow_keep(std::size_t new_length) {
            if(new_length <= length) {
                return;
            }

            DeviceArray newmem(new_length);
            if(length != 0) {
                throw_if_cuda_error("Could not move buffer on device",
                                    cudaMemcpy(newmem.mem_on_device.get(), mem_on_device.get(), length * sizeof(T), cudaMemcpyDeviceToDevice));
                auto mv_begin = std::make_move_iterator(mem_on_host.get());
                auto mv_end = mv_begin + length;
                std::copy(mv_begin, mv_end, newmem.mem_on_host.get());
            }
            *this = std::move(newmem);
        }

    private:
        std::size_t length;
        std::unique_ptr<T[]> mem_on_host;
        std::unique_ptr<T[], CUDAFreeDeleter> mem_on_device;

        IVARP_H static T* alloc_device_ptr(std::size_t length) {
            void* ptr;
            auto errcode = cudaMalloc(&ptr, length * sizeof(T));
            if(errcode != cudaSuccess) {
                throw CudaError("Could not allocate device memory", errcode);
            }
            return static_cast<T*>(ptr);
        }
    };
}
}
