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

#if defined(__CUDA_ARCH__)

namespace ivarp {
    template<typename T,
             std::enable_if_t<std::is_unsigned<T>::value && sizeof(T) == sizeof(unsigned), int> = 0>
    IVARP_D static inline T gpu_atomic_add(T* add_to, T value) noexcept {
        return static_cast<T>(
            atomicAdd(reinterpret_cast<unsigned*>(reinterpret_cast<char*>(add_to)), static_cast<unsigned>(value))
        );
    }

    template<typename T,
             std::enable_if_t<std::is_unsigned<T>::value &&
                              sizeof(T) == sizeof(unsigned long long int) &&
                              sizeof(unsigned) != sizeof(unsigned long long), int> = 0>
    IVARP_D static inline T gpu_atomic_add(T* add_to, T value) noexcept {
        return static_cast<T>(
            atomicAdd(reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(add_to)),
                      static_cast<unsigned long long>(value))
        );
    }

    static inline IVARP_D std::size_t gpu_atomic_add(std::size_t* add_to, std::size_t value) noexcept {
        return gpu_atomic_add<std::size_t>(add_to, value);
    }
}

#endif
