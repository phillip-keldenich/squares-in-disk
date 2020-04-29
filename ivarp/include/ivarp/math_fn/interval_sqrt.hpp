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
// Created by Phillip Keldenich on 30.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// For rational intervals, square roots are implemented in the library.
    extern IVARP_H IVARP_EXPORTED_SYMBOL IRational rational_interval_sqrt(const IRational &x, unsigned precision);

#if !defined(__llvm__) && !defined(__GNUC__) && defined(_MSC_VER)
    extern __declspec(dllimport) float  do_sqrt_rd(float x) noexcept;
    extern __declspec(dllimport) double do_sqrt_rd(double x) noexcept;
    extern __declspec(dllimport) float  do_sqrt_ru(float x) noexcept;
    extern __declspec(dllimport) double do_sqrt_ru(double x) noexcept;
#endif

    /// Do the actual rounding square roots for floats/doubles.
    /// There is no way to use the opposite trick here; furthermore, we have
    /// to take care to prevent the compiler (clang is particularly bad in that regard)
    /// from moving the rounding-mode change across the sqrt operations.
    /// The choice is between either terrible or even incorrect code and inline assembly.
    static inline IVARP_HD IFloat builtin_interval_do_sqrt(float lb, float ub, bool undefined) {
#if defined(__CUDA_ARCH__)
        lb = __fsqrt_rd(lb);
        ub = __fsqrt_ru(ub);
#elif defined(__llvm__) || defined(__GNUC__)
        std::uint32_t fpmode;
        asm volatile(
            "sqrtss %0, %0\n"    // round-down sqrt
            "stmxcsr %2\n"       // read control register (%2 must be memory)
            "xorl $0x6000, %2\n" // swap from round-down to round-up
            "ldmxcsr %2\n"       // change control register
            "sqrtss %1, %1\n"    // round-up sqrt
            "xorl $0x6000, %2\n" // reset to round-down
            "ldmxcsr %2\n"       // restore control register
            : "+x"(lb), "+x"(ub), "=m"(fpmode)
        );
#elif defined(_MSC_VER)
        // no inline assembly for MSVC 64 bit, but in "Release" mode,
        // the compiler breaks the code if we use intrinsics. So we use masm.
        lb = do_sqrt_rd(lb);
        ub = do_sqrt_ru(ub);
#else
#error "FIXME: Missing assembly!"
#endif
        return IFloat{lb, ub, undefined};
    }

    static inline IVARP_HD IDouble builtin_interval_do_sqrt(double lb, double ub, bool undefined) {
#if defined(__CUDA_ARCH__)
        lb = __dsqrt_rd(lb);
        ub = __dsqrt_ru(ub);
#elif defined(__llvm__) || defined(__GNUC__)
        std::uint32_t fpmode;
        asm volatile(
            "sqrtsd %0, %0\n"    // round-down sqrt
            "stmxcsr %2\n"       // read control register (%2 must be memory)
            "xorl $0x6000, %2\n" // swap from round-down to round-up
            "ldmxcsr %2\n"       // change control register
            "sqrtsd %1, %1\n"    // round-up sqrt
            "xorl $0x6000, %2\n" // reset to round-down
            "ldmxcsr %2\n"       // restore control register
            : "+x"(lb), "+x"(ub), "=m"(fpmode)
        );
#elif defined(_MSC_VER)
        // no inline assembly for MSVC 64 bit, but in "Release" mode,
        // the compiler breaks the code if we use intrinsics
        lb = do_sqrt_rd(lb);
        ub = do_sqrt_ru(ub);
#else
#error "FIXME: Missing assembly!"
#endif
        return IDouble{lb, ub, undefined};
    }

    /// Evaluation of square roots for intervals of builtin types.
    /// No opposite trick here; we have to change the rounding mode to up and back.
    template<typename BoundsType, typename FloatType> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, Interval<FloatType>>
            builtin_interval_sqrt(Interval<FloatType> x) noexcept
    {
        FloatType lb = x.lb();
        FloatType ub = x.ub();
        bool und = x.possibly_undefined();
        if(!fixed_point_bounds::nonnegative<BoundsType>(x)) {
            if(ub < 0) {
                return Interval<FloatType>{ -infinity, infinity, true };
            }
            lb = 0;
            und = true;
        }
        return builtin_interval_do_sqrt(lb, ub, und);
    }
}

    /// Overloaded sqrt evaluation function that forwards to the corresponding functions for the given number type.
    template<typename Context, typename FloatType> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, FloatType> sqrt(FloatType x) noexcept
    {
        return IVARP_NOCUDA_USE_STD sqrt(x);
    }

    template<typename Context, typename FloatType> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, Interval<FloatType>>
            sqrt(const Interval<FloatType>& x) noexcept
    {
        return impl::builtin_interval_sqrt<fixed_point_bounds::Unbounded>(x);
    }

    template<typename Context, typename BoundsType, typename NumberType> static inline IVARP_HD
        std::enable_if_t<IsIntervalType<NumberType>::value && AllowsCUDA<NumberType>::value, NumberType>
            bounded_sqrt(const NumberType& x) noexcept
    {
        // pass bounds to interval sqrt method
        return impl::builtin_interval_sqrt<BoundsType>(x);
    }

    template<typename Context> static inline IVARP_H IRational sqrt(const IRational& r) {
        return impl::rational_interval_sqrt(r, Context::irrational_precision);
    }

    template<typename Context, typename BoundsType, typename NumberType> static inline IVARP_H
        std::enable_if_t<!IsIntervalType<NumberType>::value || !AllowsCUDA<NumberType>::value, NumberType>
            bounded_sqrt(const NumberType& x) noexcept
    {
        // discard bounds
        return ::ivarp::template sqrt<Context>(x);
    }
}
