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
// Created by Phillip Keldenich on 2019-10-04.
//

#include "ivarp/number.hpp"
#include <cmath>

namespace ivarp {
namespace impl {
    static const double max_float = max_value<float>();

    static IVARP_HD float float_ub_from_double(double d) {
        if(d > max_float) {
            return inf_value<float>();
        } else if(d < -max_float) {
            return -std::numeric_limits<float>::max();
        } else {
            float res = static_cast<float>(d);
            double dres = res;
            if(dres < d) {
                res = std::nextafterf(res, std::numeric_limits<float>::infinity());
            }
            return res;
        }
    }

    static float float_lb_from_double(double d) {
        if(d > max_float) {
            return std::numeric_limits<float>::max();
        } else if(d < -max_float) {
            return -std::numeric_limits<float>::infinity();
        } else {
            float res = static_cast<float>(d);
            double dres = res;
            if(dres > d) { // LCOV_EXCL_LINE
                // this is just here in case the compiler breaks something w.r.t. rounding.
                res = std::nextafterf(res, -std::numeric_limits<float>::infinity()); // LCOV_EXCL_LINE -
            } // LCOV_EXCL_LINE
            return res;
        }
    }

    IFloat ifloat_from_double(double d) {
        return {float_lb_from_double(d), float_ub_from_double(d)};
    }

    IFloat ifloat_from_rational(const Rational& r) {
        return ConvertTo<IFloat>{}.convert(idouble_from_rational(r));
    }

    IDouble idouble_from_rational(const Rational& r) {
        double d = r.get_d();
        if(r == d) {
            // exactly representable as double
            return IDouble{d};
        } else {
            // GMP rounded this value towards zero
            if(d == 0) {
                if(r < 0) {
                    return IDouble{-std::numeric_limits<double>::min(), 0.0};
                } else {
                    return IDouble{0.0, std::numeric_limits<double>::min()};
                }
            } else if(d > 0) {
                return IDouble{d, std::nextafter(d, std::numeric_limits<double>::infinity())};
            } else {
                return IDouble{std::nextafter(d, -std::numeric_limits<double>::infinity()), d};
            }
        }
    }
}
}
