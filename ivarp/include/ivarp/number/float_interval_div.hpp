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
namespace impl {
    /// Perform float/double division for interval arithmetic in the common case of all bounds being finite.
    static inline IVARP_HD Bounds<float> ia_div(float lbnum, float lbden, float ubnum, float ubden) noexcept {
#ifdef __CUDA_ARCH__
        return {__fdiv_rd(lbnum, lbden), __fdiv_ru(ubnum,ubden)};
#else
        __m128 num = _mm_set_ps(lbnum, opacify(-ubnum), 0.f, 0.f);
        __m128 den = _mm_set_ps(lbden, opacify(ubden), 1.f, 1.f);
        __m128 res = _mm_div_ps(num, den); // NOLINT
        alignas(__m128) float rf[4];
        _mm_store_ps(rf, res);
        return {rf[3], -rf[2]};
#endif
    }

    static inline IVARP_HD Bounds<double> ia_div(double lbnum, double lbden, double ubnum, double ubden) noexcept {
#ifdef __CUDA_ARCH__
        return {__ddiv_rd(lbnum, lbden), __ddiv_ru(ubnum,ubden)};
#else
        __m128d num = _mm_set_pd(lbnum, opacify(-ubnum));
        __m128d den = _mm_set_pd(lbden, opacify(ubden));
        __m128d res = _mm_div_pd(num, den); // NOLINT
        alignas(__m128d) double rd[2];
        _mm_store_pd(rd, res);
        return {rd[1], -rd[0]};
#endif
    }

    static inline IVARP_HD float div_rd(float num, float den) noexcept {
#ifdef __CUDA_ARCH__
        return __fdiv_rd(num, den);
#else
        return opacify(opacify(num) / den);
#endif
    }

     static inline IVARP_HD double div_rd(double num, double den) noexcept {
#ifdef __CUDA_ARCH__
        return __ddiv_rd(num, den);
#else
        return opacify(opacify(num) / den);
#endif
    }

    static inline IVARP_HD float
        div_ru(float num, float den) noexcept
    {
#ifdef __CUDA_ARCH__
        return __fdiv_ru(num, den);
#else
        return opacify(-(opacify(-num) / den));
#endif
    }

    static inline IVARP_HD double
        div_ru(double num, double den) noexcept
    {
#ifdef __CUDA_ARCH__
        return __ddiv_ru(num, den);
#else
        return opacify(-(opacify(-num) / den));
#endif
    }
}
}

#include "interval_div.hpp"

namespace ivarp {
    template<> inline Interval<float> &Interval<float>::operator/=(IFloat o_) noexcept {
        impl::ia_div<float>(*this, o_);
	    return *this;
	}

	template<> inline Interval<double> &Interval<double>::operator/=(IDouble o_) noexcept {
	    impl::ia_div<double>(*this, o_);
	    return *this;
	}
}
