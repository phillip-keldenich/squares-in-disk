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

#include "ivarp/math_fn.hpp"

namespace ivarp {
namespace impl {
    template<bool RoundUp> static Rational rational_sqrt(const Rational& v, unsigned precision) {
        DynamicMPFRNumber x(precision);
        mpfr_set_q(x, v.get_mpq_t(), RoundUp ? MPFR_RNDU : MPFR_RNDD);
        mpfr_sqrt(x, x, RoundUp ? MPFR_RNDU : MPFR_RNDD);
        Rational result;
        mpfr_get_q(result.get_mpq_t(), x);
        return result;
    }
}
}

auto ivarp::impl::rational_interval_sqrt(const IRational& x, unsigned precision) -> IRational {
    IRational result{0, 0, x.possibly_undefined() };
    if(BOOST_UNLIKELY(!x.finite_lb() || x.lb() < 0)) {
        if(x.ub() < 0) {
            return IRational{-infinity, infinity, true};
        }
        result.set_undefined(true);
    } else {
        result.set_lb(rational_sqrt<false>(x.lb(), precision));
    }
    if(!x.finite_ub()) {
        result.set_ub(infinity);
    } else {
        result.set_ub(rational_sqrt<true>(x.ub(), precision));
    }
    return result;
}
