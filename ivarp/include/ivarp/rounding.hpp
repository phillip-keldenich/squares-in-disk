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
// Created by Phillip Keldenich on 2019-09-23.
//

#pragma once

#include <cfloat>
#include <cfenv>
#include <cstdlib>
#include <cmath>
#include <boost/predef.h>

#include "ivarp/cuda.hpp"

// we need to access the rounding mode.
// a prover program changes the rounding mode to round-down globally when a proof starts;
// rounding mode switches are then restricted to operations that cannot use the opposite trick.
// changing the rounding mode may break things with a lot of compiler optimizations.
// the biggest problems in that regard are compile-time constant folding (if it goes wrong),
// reordering of instructions across rounding-mode changes, and omission of "equivalent" computations that
// are not equivalent due to a changed rounding mode.
// GCC needs -frounding-math. Clang must deal without it, as it does not support the flag at all.
// Across all compilers we support, CLang is by far the least reliable w.r.t. rounding modes; the opacify() function should take care of this.
// MSVC needs /fp:strict but still sometimes breaks things with constant folding; it also lacks an inline assembler that could be used to work around bugs like this.
// Thus, we have to resort to actual assembly (yuck!).
// Moreover, we have to get rid of extended precision and double-rounding and other problems occurring with the x87 FPU;
// we enforce use of sse2 which does not have extended precision to get rid of these issues.

// Check that these flags worked:
#ifdef __FLT_EVAL_METHOD__
#define IVARP_FLT_EVAL_METHOD __FLT_EVAL_METHOD__
#endif
#if !defined(IVARP_FLT_EVAL_METHOD) && defined(FLT_EVAL_METHOD)
#define IVARP_FLT_EVAL_METHOD FLT_EVAL_METHOD
#endif

#ifndef IVARP_FLT_EVAL_METHOD
#error "Could not detect floating-point evaluation method to ensure there are no double-rounding issues!"
#endif

#if IVARP_FLT_EVAL_METHOD != 0 && IVARP_FLT_EVAL_METHOD != 1
#error "Could not ensure that there are no double-rounding issues!"
#elif IVARP_FLT_EVAL_METHOD != 0
#warning "float and double evaluation both use double precision!"
#endif
#undef IVARP_FLT_EVAL_METHOD // don't clutter the global namespace.

#if defined(_MSC_VER) && !defined(_M_FP_STRICT) // for MSVC, check the strict flag.
#error "MSVC is not using strict floating-point mode (/fp:strict)!"
#endif

#if defined(_MSC_VER) && !defined(__llvm__)
#pragma fenv_access (on)
#endif

// SSE2 detection for CLang & GCC.
#if defined(__GNUC__)
#ifndef __SSE2_MATH__
#error "Could not determine whether we are using SSE2!"
#endif
#endif

/// IVARP_ENSURE_ATLOAD_ROUNDDOWN: If the rounding mode is not set to round-down at global initialization time,
/// set it to round-down for the scope of the function in which this macro is invoked.
#if defined(_WIN32) || defined(__WIN32__)
#define IVARP_ENSURE_ATLOAD_ROUNDDOWN() ::ivarp::SetRoundDown ivarp_ensure_rounddown_
#else
#define IVARP_ENSURE_ATLOAD_ROUNDDOWN() (static_cast<void>(0))
#endif

// SSE2 detection for MSVC.
#if defined(_MSC_VER) && !defined(_M_X64) // 64-bit always uses at least SSE2.
#ifndef _M_IX86_FP
#error "Could not determine whether we are using SSE2!"
#elif _M_IX86_FP < 2
#error "Compilation does not use (at least) SSE2!"
#endif
#endif

#if defined(__GNUC__)
#include <xmmintrin.h>
#else
#include <emmintrin.h>
#endif

#if defined(_MM_FUNCTIONALITY)
#error "SSE2 intrinsics actually use a fallback implementation based on the x87 FPU."
#endif

#include "ivarp/symbol_export.hpp"

namespace ivarp {
    static inline double opacify(double d) noexcept;
    static inline void do_opacify(double& d) noexcept;
    static inline float opacify(float f) noexcept;
    static inline void do_opacify(float& f) noexcept;

    class SetRound {
    protected:
        static constexpr int invalid_rounding_mode = -42;
        static_assert(invalid_rounding_mode != FE_DOWNWARD, "invalid_rounding_mode is not invalid!");
        static_assert(invalid_rounding_mode != FE_TONEAREST, "invalid_rounding_mode is not invalid!");
        static_assert(invalid_rounding_mode != FE_UPWARD, "invalid_rounding_mode is not invalid!");
        static_assert(invalid_rounding_mode != FE_TOWARDZERO, "invalid_rounding_mode is not invalid!");

        explicit SetRound() noexcept :
            m_prev(invalid_rounding_mode)
        {}

        explicit SetRound(int m) noexcept :
            m_prev(std::fegetround())
        {
            std::fesetround(m);
        }

        ~SetRound() {
            if(m_prev != invalid_rounding_mode) {
                std::fesetround(m_prev);
            }
        }

        int m_prev;
    };

    class SetRoundDown : private SetRound {
    public:
        explicit SetRoundDown() noexcept :
            SetRound(FE_DOWNWARD)
        {}

        SetRoundDown(SetRoundDown&& o) noexcept :
            SetRound()
        {
            m_prev = o.m_prev;
            o.m_prev = invalid_rounding_mode;
        }

        SetRoundDown(const SetRoundDown&) = delete;
        SetRoundDown &operator=(const SetRoundDown&) = delete;
        SetRoundDown &operator=(SetRoundDown&&) = delete;
        ~SetRoundDown() = default;
    };

    class SetRoundUp : private SetRound {
    public:
        explicit SetRoundUp() noexcept :
            SetRound(FE_UPWARD)
        {}

        SetRoundUp(SetRoundUp&& o) noexcept :
            SetRound()
        {
            m_prev = o.m_prev;
            o.m_prev = invalid_rounding_mode;
        }

        SetRoundUp(const SetRoundUp&) = delete;
        SetRoundUp &operator=(const SetRoundUp&) = delete;
        SetRoundUp &operator=(SetRoundUp&&) = delete;
        ~SetRoundUp() = default;
    };
}

void ivarp::do_opacify(float &f) noexcept {
#if BOOST_COMP_GNUC && !defined(IVARP_FORCE_OPACIFY)
    (void)f;
#elif BOOST_COMP_CLANG
    asm volatile("" : "+mx"(f));
#else
    volatile float f_ = f;
    f = f_;
#endif
}

void ivarp::do_opacify(double &d) noexcept {
#if BOOST_COMP_GNUC && !defined(IVARP_FORCE_OPACIFY)
    (void)d;
#elif BOOST_COMP_CLANG
    asm volatile("" : "+mx"(d));
#else
    volatile double d_ = d;
    d = d_;
#endif
}

float ivarp::opacify(float f) noexcept {
    do_opacify(f);
    return f;
}

double ivarp::opacify(double d) noexcept {
    do_opacify(d);
    return d;
}
