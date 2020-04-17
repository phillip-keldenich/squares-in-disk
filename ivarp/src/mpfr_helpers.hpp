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
// Created by Phillip Keldenich on 22.10.19.
//

#pragma once

#include <mpfr.h>

namespace ivarp {
namespace impl {
    /// Overloaded function to set mpfr variables to floating-point values.
    static inline void mpfr_set_fltp(mpfr_ptr target, float x, mpfr_rnd_t round) {
        mpfr_set_flt(target, x, round);
    }
    static inline void mpfr_set_fltp(mpfr_ptr target, double x, mpfr_rnd_t round) {
        mpfr_set_d(target, x, round);
    }

    /// Template function to get floating-point values from mpfr variables.
    template<typename F> inline F mpfr_get_fltp(mpfr_srcptr src, mpfr_rnd_t round);
    template<> inline float mpfr_get_fltp<float>(mpfr_srcptr src, mpfr_rnd_t round) {
        return mpfr_get_flt(src, round);
    }
    template<> inline double mpfr_get_fltp<double>(mpfr_srcptr src, mpfr_rnd_t round) {
        return mpfr_get_d(src, round);
    }

    /// Perform a unary MPFR operation, rounding up or down.
    template<int MPFR_OP(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), bool RoundUp>
        static inline Rational round_mpfr_op_rational(const Rational& r, unsigned precision)
    {
        DynamicMPFRNumber mx_lb(precision);
        DynamicMPFRNumber mx_ub(precision);
        mpfr_set_q(mx_lb, r.get_mpq_t(), MPFR_RNDD);
        mpfr_set_q(mx_ub, r.get_mpq_t(), MPFR_RNDU);
        MPFR_OP(mx_lb, mx_lb, RoundUp ? MPFR_RNDU : MPFR_RNDD);
        MPFR_OP(mx_ub, mx_ub, RoundUp ? MPFR_RNDU : MPFR_RNDD);
        if(RoundUp) {
            mpfr_max(mx_lb, mx_lb, mx_ub, MPFR_RNDU);
        } else {
            mpfr_min(mx_lb, mx_lb, mx_ub, MPFR_RNDN);
        }
        Rational res;
        mpfr_get_q(res.get_mpq_t(), mx_lb);
        return res;
    }
}
}
