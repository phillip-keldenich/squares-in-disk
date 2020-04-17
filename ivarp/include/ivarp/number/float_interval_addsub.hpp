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

/**
 * @file float_interval_addsub.hpp
 * Addition and subtraction for float or double intervals, for both CPU and CUDA.
 */

namespace ivarp {
        template<> inline IFloat& Interval<float>::operator+=(IFloat o_) noexcept {
#ifdef __CUDA_ARCH__
	    m_lb = __fadd_rd(m_lb, o_.m_lb);
	    m_ub = __fadd_ru(m_ub, o_.m_ub);
#else
        // Apparently, GCC does not need opacify here, but clang does; otherwise, incorrect code is generated.
        __m128 t = _mm_set_ps(m_lb, opacify(-m_ub), 0.f, 0.f);
        __m128 o = _mm_set_ps(o_.m_lb, opacify(-o_.m_ub), 0.f, 0.f);
        __m128 r = _mm_add_ps(t, o); // NOLINT
        alignas(__m128) float rf[4];
        _mm_store_ps(rf, r);
        m_lb = rf[3];
        m_ub = -rf[2];
#endif
        m_undefined |= o_.m_undefined;
        return *this;
    }

    template<> inline IDouble& Interval<double>::operator+=(IDouble o_) noexcept {
#ifdef __CUDA_ARCH__
	    m_lb = __dadd_rd(m_lb, o_.m_lb);
	    m_ub = __dadd_ru(m_ub, o_.m_ub);
#else
        __m128d t = _mm_set_pd(m_lb, opacify(-m_ub));
        __m128d o = _mm_set_pd(o_.m_lb, opacify(-o_.m_ub));
        __m128d r = _mm_add_pd(t, o); // NOLINT
         alignas(__m128d) double rd[2];
        _mm_store_pd(rd, r);
        m_lb = rd[1];
        m_ub = -rd[0];
#endif
        m_undefined |= o_.m_undefined;
        return *this;
    }

    template<> inline IFloat& Interval<float>::operator-=(IFloat o_) noexcept {
#ifdef __CUDA_ARCH__
        m_lb = __fsub_rd(m_lb, o_.m_ub);
        m_ub = __fsub_ru(m_ub, o_.m_lb);
#else
        __m128 t = _mm_set_ps(m_lb, opacify(-m_ub), 0.f, 0.f);
        __m128 o = _mm_set_ps(o_.m_ub, opacify(-o_.m_lb), 0.f, 0.f);
        __m128 r = _mm_sub_ps(t, o); // NOLINT
        alignas(__m128) float rf[4];
        _mm_store_ps(rf, r);
        m_lb = rf[3];
        m_ub = -rf[2];
#endif
        m_undefined |= o_.m_undefined;
        return *this;
    }

    template<> inline IDouble& Interval<double>::operator-=(IDouble o_) noexcept {
#ifdef __CUDA_ARCH__
	    m_lb = __dsub_rd(m_lb, o_.m_ub);
        m_ub = __dsub_ru(m_ub, o_.m_lb);
#else
        __m128d t = _mm_set_pd(m_lb, opacify(-m_ub));
        __m128d o = _mm_set_pd(o_.m_ub, opacify(-o_.m_lb));
        __m128d r = _mm_sub_pd(t, o); // NOLINT
        alignas(__m128d) double rd[2];
        _mm_store_pd(rd, r);
        m_lb = rd[1];
        m_ub = -rd[0];
#endif
        m_undefined |= o_.m_undefined;
        return *this;
    }
}
