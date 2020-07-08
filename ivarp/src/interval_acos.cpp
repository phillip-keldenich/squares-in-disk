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
// Created by Phillip Keldenich on 26.11.19.
//

#include "ivarp/math_fn.hpp"
#include "mpfr_helpers.hpp"

template<bool RoundUp, typename FloatType> static inline FloatType rounded_acos(FloatType x) noexcept {
    MPFR_DECL_INIT(mx, std::numeric_limits<FloatType>::digits); // NOLINT
    ivarp::impl::mpfr_set_fltp(mx, x, RoundUp ? MPFR_RNDD : MPFR_RNDU);
    mpfr_acos(mx, mx, RoundUp ? MPFR_RNDU : MPFR_RNDD);
    return ivarp::impl::mpfr_get_fltp<FloatType>(mx, RoundUp ? MPFR_RNDU : MPFR_RNDD);
}

/// Implementation of rounded arc cosine for rationals.
template<bool RoundUp> static inline ivarp::Rational rounded_acos(const ivarp::Rational& r, unsigned precision) {
    return ivarp::impl::round_mpfr_op_rational<mpfr_acos, RoundUp>(r, precision);
}

ivarp::IRational ivarp::impl::rational_interval_acos(const ivarp::IRational &x, unsigned precision) {
    if(BOOST_UNLIKELY(!x.finite_lb() || !x.finite_ub() || x.lb() < -1 || x.ub() > 1)) {
        ivarp::IRational tmp{!x.finite_lb() || x.lb() < -1 ? -1 : x.lb(),
                             !x.finite_ub() || x.ub() > 1 ? 1 : x.ub()};
        return { rounded_acos<false>(tmp.ub(), precision), rounded_acos<true>(tmp.lb(), precision), true};
    } else {
        return { rounded_acos<false>(x.ub(), precision), rounded_acos<true>(x.lb(), precision),
                 x.possibly_undefined()};
    }
}

IVARP_H float ivarp::impl::cpu_acos_rd(float x) noexcept {
    return rounded_acos<false, float>(x);
}

IVARP_H float ivarp::impl::cpu_acos_ru(float x) noexcept {
    return rounded_acos<true, float>(x);
}

IVARP_H double ivarp::impl::cpu_acos_rd(double x) noexcept {
    return rounded_acos<false, double>(x);
}

IVARP_H double ivarp::impl::cpu_acos_ru(double x) noexcept {
    return rounded_acos<true, double>(x);
}

