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

#include "sincos_period_info.hpp"
#include "mpfr_helpers.hpp"

namespace ivarp {
namespace impl {
    /// Implementation of rounded sine for nonnegative floating-point values.
    template<bool RoundUp, typename F> static inline
        std::enable_if_t<std::is_floating_point<F>::value, F> round_sin(F x, unsigned /*precision_ignored*/)
    {
        MPFR_DECL_INIT(mx, std::numeric_limits<F>::digits); // NOLINT
        mpfr_set_fltp(mx, x, RoundUp ? MPFR_RNDU : MPFR_RNDD); // rounding should only occur for subnormals
        //                                                               (i.e. very close to zero, where
        //                                                               rounddown is correct if we want to round down)
        mpfr_sin(mx, mx, RoundUp ? MPFR_RNDU : MPFR_RNDD);
        return mpfr_get_fltp<F>(mx, RoundUp ? MPFR_RNDU : MPFR_RNDD);
    }

    /// Implementation of rounded sine for nonnegative rational values.
    template<bool RoundUp> static inline Rational round_sin(const Rational& r, unsigned precision) {
        return round_mpfr_op_rational<mpfr_sin, RoundUp>(r, precision);
    }

    /// Implementation of interval sine for intervals that do not wrap across a multiple of 2pi.
    template<typename NumberType> static inline Interval<NumberType>
        interval_sin_nowrap(const PeriodInfo<NumberType>& period, Interval<NumberType> x, unsigned precision)
    {
        if(period.lb_period_flt <= 0.25) {
            if(period.ub_period_flt < 0.25) {
                return Interval<NumberType>{
                        round_sin<false>(x.lb(), precision),
                        round_sin<true>(x.ub(), precision), x.possibly_undefined()
                };
            } else if(period.ub_period_flt < 0.75) {
                return Interval<NumberType>{
                        std::min(round_sin<false>(x.lb(), precision), round_sin<false>(x.ub(), precision)),
                        NumberType(1), x.possibly_undefined()
                };
            } else {
                return Interval<NumberType>{NumberType(-1), NumberType(1), x.possibly_undefined()};
            }
        } else {
            if(period.ub_period_flt < 0.75) {
                return Interval<NumberType>{
                        round_sin<false>(x.ub(), precision),
                        round_sin<true>(x.lb(), precision),
                        x.possibly_undefined()
                };
            } else if(period.lb_period_flt <= 0.75) {
                return Interval<NumberType>{
                        NumberType(-1),
                        std::max(round_sin<true>(x.lb(), precision), round_sin<true>(x.ub(), precision)),
                        x.possibly_undefined()
                };
            } else {
                return Interval<NumberType>{
                        round_sin<false>(x.lb(), precision),
                        round_sin<true>(x.ub(), precision),
                        x.possibly_undefined()
                };
            }
        }
    }

    /// Implementation of interval sine for intervals for which the lower bound is one 2pi period before the upper bound.
    template<typename NumberType> static inline Interval<NumberType>
        interval_sin_wrap(const PeriodInfo<NumberType>& period, Interval<NumberType> x, unsigned precision)
    {
        if(period.lb_period_flt <= 0.25) {
            return Interval<NumberType>{NumberType(-1), NumberType(1), x.possibly_undefined()};
        } else if(period.lb_period_flt <= 0.75) {
            if(period.ub_period_flt < 0.25) {
                return Interval<NumberType>{
                        NumberType(-1),
                        std::max(round_sin<true>(x.lb(), precision), round_sin<true>(x.ub(), precision)),
                        x.possibly_undefined()
                };
            } else {
                return Interval<NumberType>{NumberType(-1), NumberType(1), x.possibly_undefined()};
            }
        } else {
            if(period.ub_period_flt < 0.25) {
                return Interval<NumberType>{
                        round_sin<false>(x.lb(), precision),
                        round_sin<true>(x.ub(), precision), x.possibly_undefined()
                };
            } else if(period.ub_period_flt < 0.75) {
                return Interval<NumberType>{
                        std::min(round_sin<false>(x.lb(), precision), round_sin<false>(x.ub(), precision)),
                        NumberType(1), x.possibly_undefined()
                };
            } else {
                return Interval<NumberType>{-1, 1, x.possibly_undefined()};
            }
        }
    }

    /// Implementation of sine for non-negative intervals.
    template<typename NumberType> static inline
        Interval<NumberType>
            interval_sin_nonnegative(const Interval<NumberType>& x, unsigned precision)
    {
        PeriodInfo<NumberType> period = get_period_info(x, precision);

        // the bounds are not definitely in adjacent periods
        if(opacify(period.lb_period_int + NumberType(1)) < period.ub_period_int) {
            return Interval<NumberType>{NumberType(-1), NumberType(1), x.possibly_undefined()};
        }

        if(period.lb_period_int == period.ub_period_int) {
            // no wrap-around
            return interval_sin_nowrap(period, x, precision);
        } else {
            // wrap-around
            return interval_sin_wrap(period, x, precision);
        }
    }

    /// Use sine's symmetry to get rid of all negative input values.
    template<typename NumberType> static inline Interval<NumberType> interval_sin_symm(const Interval<NumberType>& x,
                                                                                       unsigned precision)
    {
        using IV = Interval<NumberType>;

        if(!is_finite(x)) {
            return Interval<NumberType>(NumberType(-1), NumberType(1), x.possibly_undefined());
        }

        if(x.ub() <= 0) {
            // negative
            return -interval_sin_nonnegative(-x, precision);
        } else if(x.lb() < 0) {
            // mixed
            IV r = -interval_sin_nonnegative(IV{0, -x.lb(), x.possibly_undefined()}, precision);
            r.do_join(interval_sin_nonnegative(IV{0, x.ub(), x.possibly_undefined()}, precision));
            return r;
        } else {
            // positive
            return interval_sin_nonnegative(x, precision);
        }
    }

    IFloat builtin_interval_sin(IFloat x) {
        return interval_sin_symm(x, 0);
    }

    IDouble builtin_interval_sin(IDouble x) {
        return interval_sin_symm(x, 0);
    }

    IRational rational_interval_sin(const IRational& x, unsigned precision) {
        return interval_sin_symm(x, precision);
    }
}
}
