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

#include "ivarp/math_fn.hpp"
#include "irr_constants.hpp"
#include "mpfr_helpers.hpp"

namespace ivarp {
namespace impl {
    /// Information about the position of an interval relative to the 2pi period of sine/cosine.
    template<typename NumberType>
    struct PeriodInfo
    {
        using IntPartType = ModfResultType<NumberType>;

        NumberType lb_period, ub_period;
        NumberType lb_period_flt, ub_period_flt;
        IntPartType lb_period_int, ub_period_int;
    };

    /// Get informtion about the position of an interval relative to the 2pi period of sine/cosine (floating-point case).
    template<typename FloatType> inline
        std::enable_if_t<std::is_floating_point<FloatType>::value, PeriodInfo<FloatType>>
            get_period_info(Interval<FloatType> x, unsigned /*ignored*/) noexcept
    {
        // we stay in round down mode here
        PeriodInfo<FloatType> info; // NOLINT
        info.lb_period = opacify(x.lb()) * opacify(FloatConstants<FloatType>::rec_two_pi.lb());
        info.ub_period = opacify(-opacify(opacify(-x.ub()) * opacify(FloatConstants<FloatType>::rec_two_pi.ub())));
        info.lb_period_flt = ivarp::modf(info.lb_period, &info.lb_period_int);
        info.ub_period_flt = ivarp::modf(info.ub_period, &info.ub_period_int);
        return info;
    }

    /// Get information about the position of an interval relative to the 2pi period of sine/cosine (rational case).
    inline PeriodInfo<Rational> get_period_info(const IRational& r, unsigned precision) {
        PeriodInfo<Rational> info;
        if(precision == default_irrational_precision) {
            info.lb_period = FloatConstants<Rational>::rec_two_pi.lb() * r.lb();
            info.ub_period = FloatConstants<Rational>::rec_two_pi.ub() * r.ub();
        } else {
            IRational r2pi = rational_rec_two_pi(precision);
            info.lb_period = r2pi.lb() * r.lb();
            info.ub_period = r2pi.ub() * r.ub();
        }
        info.lb_period_flt = ivarp::modf(info.lb_period, &info.lb_period_int);
        info.ub_period_flt = ivarp::modf(info.ub_period, &info.ub_period_int);
        return info;
    }
}
}
