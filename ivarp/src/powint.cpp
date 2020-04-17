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
// Created by Phillip Keldenich on 09.10.19.
//

#include "ivarp/math_fn.hpp"
#include "ivarp/rounding.hpp"

namespace ivarp {
namespace impl {
    Rational rational_ipow(const Rational& r,  unsigned p) {
        Rational x = r;
        Rational y = (p % 2) ? r : 1;
        for(p /= 2; p > 0; p /= 2) {
            x *= x;
            if(p % 2) { y *= x; }
        }
        return y;
    }

    static IRational rational_ipow_monotonic(const IRational& r, unsigned p) {
        return {rational_ipow(r.lb(), p), rational_ipow(r.ub(), p), r.possibly_undefined()};
    } // LCOV_EXCL_LINE

    static IRational rational_ipow_even(const IRational& r, unsigned p) {
        if(!is_finite(r)) {
            if(p == 0) {
                return IRational{1,1, r.possibly_undefined() };
            } else {
                return IRational{0, infinity, r.possibly_undefined()};
            }
        }

        if(r.ub() < 0) {
            // negative
            return rational_ipow_monotonic(-r, p);
        }

        if(r.lb() < 0) {
            // mixed sign
            Rational alb = -r.lb();
            if(alb < r.ub()) {
                alb = r.ub();
            }
            return rational_ipow_monotonic(IRational{Rational{0}, alb, r.possibly_undefined()}, p);
        }

        return rational_ipow_monotonic(r, p);
    }

    static IRational rational_ipow_odd(const IRational& r, unsigned p) {
        if(BOOST_LIKELY(is_finite(r))) {
            return rational_ipow_monotonic(r, p);
        }

        IRational result{0,0, r.possibly_undefined()};
        if(!r.finite_lb()) {
            result.set_lb(-infinity);
        } else {
            result.set_lb(rational_ipow(r.lb(), p));
        }
        if(!r.finite_ub()) {
            result.set_ub(infinity);
        } else {
            result.set_ub(rational_ipow(r.ub(), p));
        }
        return result;
    }

    IRational rational_ipow(const IRational& r, unsigned p) {
        if(p % 2 == 0) {
            return rational_ipow_even(r, p);
        } else {
            return rational_ipow_odd(r, p);
        }
    }
}
}
