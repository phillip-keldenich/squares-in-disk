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
// Created by Phillip Keldenich on 18.11.19.
//

#include <atomic>
#include <type_traits>

namespace ivarp {
    template<typename FloatType>
        static inline IVARP_H std::enable_if_t<std::is_floating_point<FloatType>::value>
            atomic_min(std::atomic<FloatType>* inout, typename NoTypeDeduction<FloatType>::Type in2) noexcept
    {
        FloatType v = inout->load();
        do {
            if(v < in2) {
                return;
            }
        } while(!inout->compare_exchange_weak(v, in2));
    }

#if defined(__CUDA_ARCH__)
    template<typename FloatType> struct FloatReplacementType;
    template<> struct FloatReplacementType<float> {
        using Type = __attribute__((may_alias)) std::int32_t;

        IVARP_D static float* to_float_ptr(Type* t) noexcept {
            return reinterpret_cast<float*>(reinterpret_cast<char*>(t));
        }

        IVARP_D static Type* to_rep_ptr(float* f) noexcept {
            return reinterpret_cast<Type*>(reinterpret_cast<char*>(f));
        }
    };
    template<> struct FloatReplacementType<double> {
        using Type = __attribute__((may_alias)) std::int64_t;

        IVARP_D static double* to_float_ptr(Type* t) noexcept {
            return reinterpret_cast<double*>(reinterpret_cast<char*>(t));
        }

        IVARP_D static Type* to_rep_ptr(double* f) noexcept {
            return reinterpret_cast<Type*>(reinterpret_cast<char*>(f));
        }
    };

    template<typename FloatType>
        static inline IVARP_D std::enable_if_t<std::is_floating_point<FloatType>::value, bool>
            atomic_cas(FloatType* ptr, FloatType& expected, FloatType value) noexcept
    {
        using Rep = typename FloatReplacementType<FloatType>::Type;
        Rep ex = *FloatReplacementType<FloatType>::to_rep_ptr(&expected);
        Rep val = *FloatReplacementType<FloatType>::to_rep_ptr(&value);
        Rep res = atomicCAS(FloatReplacementType<FloatType>::to_rep_ptr(ptr), ex, val);
        expected = *FloatReplacementType<FloatType>::to_float_ptr(&res);
        return res == ex;
    }
#endif

    template<typename FloatType>
        static inline IVARP_D std::enable_if_t<std::is_floating_point<FloatType>::value>
            atomic_min(FloatType* inout, typename NoTypeDeduction<FloatType>::Type in2) noexcept
    {
#if IVARP_CUDA_DEVICE_CODE
        FloatType value = *inout;
        do {
            if(value <= in2) {
                return;
            }
        } while(!atomic_cas(inout, value, in2));
#endif
        IVARP_CUDA_STATIC_ASSERT_DEVICE_ONLY(FloatType);
    }

    template<typename FloatType>
        static inline IVARP_H std::enable_if_t<std::is_floating_point<FloatType>::value>
            atomic_max(std::atomic<FloatType>* inout, typename NoTypeDeduction<FloatType>::Type in2) noexcept
    {
        FloatType v = inout->load();
        do {
            if(v > in2) {
                return;
            }
        } while(!inout->compare_exchange_weak(v, in2));
    }

    template<typename FloatType>
        static inline IVARP_D std::enable_if_t<std::is_same<FloatType,float>::value>
            atomic_max(FloatType* inout, typename NoTypeDeduction<FloatType>::Type in2) noexcept
    {
#if IVARP_CUDA_DEVICE_CODE
        FloatType value = *inout;
        do {
            if(value >= in2) {
                return;
            }
        } while(!atomic_cas(inout, value, in2));
#endif
        IVARP_CUDA_STATIC_ASSERT_DEVICE_ONLY(FloatType);
    }
}
