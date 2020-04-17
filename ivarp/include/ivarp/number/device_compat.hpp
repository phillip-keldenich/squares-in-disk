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
// Created by Phillip Keldenich on 29.11.19.
//

#pragma once

namespace ivarp {
    /// Detection for CUDA-compatible types.
    template<typename T> struct AllowsCudaImpl : std::false_type {};
    template<typename N> class Interval;
    class IBool;

#define IVARP_BT_AC(t) template<> struct AllowsCudaImpl<t> : std::true_type {}

    IVARP_BT_AC(char);
    IVARP_BT_AC(unsigned char);
    IVARP_BT_AC(bool);
    IVARP_BT_AC(short);
    IVARP_BT_AC(unsigned short);
    IVARP_BT_AC(int);
    IVARP_BT_AC(unsigned);
    IVARP_BT_AC(long);
    IVARP_BT_AC(unsigned long);
    IVARP_BT_AC(long long);
    IVARP_BT_AC(unsigned long long);
    IVARP_BT_AC(float);
    IVARP_BT_AC(double);
    IVARP_BT_AC(Interval<float>);
    IVARP_BT_AC(Interval<double>);
    IVARP_BT_AC(IBool);

#undef IVARP_BT_AC

    template<typename T> using AllowsCuda = AllowsCudaImpl<BareType<T>>;
    template<typename... Args> using AllAllowCuda = AllOf<AllowsCuda<Args>::value...>;

#ifdef __CUDA_ARCH__
#define IVARP_NOCUDA_USE_STD ::
#define IVARP_CUDA_DEVICE_OR_CONSTEXPR __device__
#define IVARP_CUDA_DEVICE_CODE 1
#define IVARP_CUDA_HOST_CODE 0
#define IVARP_CUDA_STATIC_ASSERT_DEVICE_ONLY(T) static_cast<void>(0)
#else
#define IVARP_NOCUDA_USE_STD std::
#define IVARP_CUDA_DEVICE_OR_CONSTEXPR constexpr
#define IVARP_CUDA_DEVICE_CODE 0
#define IVARP_CUDA_HOST_CODE 1
#define IVARP_CUDA_STATIC_ASSERT_DEVICE_ONLY(T) static_assert(std::is_void<T>::value && !std::is_void<T>::value,\
                                                              "This code function should only be used on CUDA devices!")
#endif

namespace impl {
    /// Device-compatible pair of numbers used as bounds.
    template<typename FloatType> struct Bounds {
        Bounds() noexcept = default;
        IVARP_HD Bounds(FloatType lb, FloatType ub) noexcept :
            lb(lb), ub(ub)
        {}

        FloatType lb, ub;
    };

    /// Device-compatible pair of Number and int used for infinities.
    template<typename Number>
        struct WithInfSign
    {
        Number number;
        int inf_sign;
    };

    static_assert(std::is_trivial<Bounds<float>>::value, "Bounds must be a trivial type!");
    static_assert(std::is_trivial<Bounds<double>>::value, "Bounds must be a trivial type!");
    static_assert(std::is_trivial<WithInfSign<float>>::value, "Bounds must be a trivial type!");
    static_assert(std::is_trivial<WithInfSign<double>>::value, "Bounds must be a trivial type!");

    /// Unfortunately, std::numeric_limits::max or ::infinity is not really supported for CUDA.
    template<typename FT> static inline IVARP_HD FT max_value() noexcept;
    template<> inline IVARP_HD float max_value<float>() noexcept {
#ifdef __CUDA_ARCH__
        return FLT_MAX;
#else
        return std::numeric_limits<float>::max();
#endif
    }
    template<> inline IVARP_HD double max_value<double>() noexcept {
#ifdef __CUDA_ARCH__
            return DBL_MAX;
#else
            return std::numeric_limits<double>::max();
#endif
    }
    template<typename FT> static inline IVARP_HD FT inf_value() noexcept;
    template<> inline IVARP_HD float inf_value<float>() noexcept {
#ifdef __CUDA_ARCH__
        return HUGE_VALF;
#else
        return std::numeric_limits<float>::infinity();
#endif
    }
    template<> inline IVARP_HD double inf_value<double>() noexcept {
#ifdef __CUDA_ARCH__
            return HUGE_VAL;
#else
            return std::numeric_limits<double>::infinity();
#endif
    }

    IVARP_HD static inline float step_up(float f) noexcept {
#ifdef __CUDA_ARCH__
        return nextafterf(f, HUGE_VALF);
#else
        return std::nextafter(f, std::numeric_limits<float>::infinity());
#endif
    }

    IVARP_HD static inline double step_up(double f) noexcept {
#ifdef __CUDA_ARCH__
        return nextafter(f, HUGE_VAL);
#else
        return std::nextafter(f, std::numeric_limits<double>::infinity());
#endif
    }

    IVARP_HD static inline float step_down(float f) noexcept {
#ifdef __CUDA_ARCH__
        return nextafterf(f, -HUGE_VALF);
#else
        return std::nextafter(f, -std::numeric_limits<float>::infinity());
#endif
    }

    IVARP_HD static inline double step_down(double f) noexcept {
#ifdef __CUDA_ARCH__
        return nextafter(f, -HUGE_VAL);
#else
        return std::nextafter(f, -std::numeric_limits<double>::infinity());
#endif
    }
}
}
