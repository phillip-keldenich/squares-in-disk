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
// Created by Phillip Keldenich on 10.10.19.
//

#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"
#include "irr_constants.hpp"

namespace ivarp {
namespace impl {
    // some constants.
    static const double lb_pi_d = 3.141592653589793115997963468544185161590576171875;
    //       < 3.14159265358979323846264338327950288 <
    static const double ub_pi_d = 3.141592653589793560087173318606801331043243408203125;
    static const double lb_2_pi_d = 6.28318530717958623199592693708837032318115234375;
    //         < 6.28318530717958647692528676655900576 <
    static const double ub_2_pi_d = 6.28318530717958712017434663721360266208648681640625;
    static const double lb_rec_2_pi_d = 0.159154943091895317852646485334844328463077545166015625;
    //             < 0.159154943091895335768883763372514362034459645740456448747 <
    static const double ub_rec_2_pi_d = 0.1591549430918953456082221009637578390538692474365234375;

    static const float lb_pi_f = 3.141592502593994140625f;
    //     < 3.14159265358979323846264338327950288 <
    static const float ub_pi_f = 3.1415927410125732421875f;
    static const float lb_2_pi_f = 6.28318500518798828125f;
    //       < 6.28318530717958647692528676655900576 <
    static const float ub_2_pi_f = 6.283185482025146484375f;
    static const float lb_rec_2_pi_f = 0.15915493667125701904296875f;
    //           < 0.159154943091895335768883763372514362034459645740456448747 <
    static const float ub_rec_2_pi_f = 0.159154951572418212890625f;

    static inline IRational default_irrational_pi() {
        MPFRNumber<default_irrational_precision> lb_mpfr_pi;
        MPFRNumber<default_irrational_precision> ub_mpfr_pi;
        mpfr_const_pi(lb_mpfr_pi, MPFR_RNDD);
        mpfr_const_pi(ub_mpfr_pi, MPFR_RNDU);
        IRational r;
        mpfr_get_q(r.lb_ref().get_mpq_t(), lb_mpfr_pi);
        mpfr_get_q(r.ub_ref().get_mpq_t(), ub_mpfr_pi);
        return r;
    }

    static inline IRational default_irrational_two_pi();
    static inline IRational default_irrational_rec_two_pi();

    const Interval<float> FloatConstants<float>::pi{lb_pi_f, ub_pi_f};
    const Interval<double>   FloatConstants<double>::pi{lb_pi_d, ub_pi_d};
    const Interval<float>    FloatConstants<float>::two_pi{lb_2_pi_f, ub_2_pi_f};
    const Interval<double>   FloatConstants<double>::two_pi{lb_2_pi_d, ub_2_pi_d};
    const Interval<float>    FloatConstants<float>::rec_two_pi{lb_rec_2_pi_f, ub_rec_2_pi_f};
    const Interval<double>   FloatConstants<double>::rec_two_pi{lb_rec_2_pi_d, ub_rec_2_pi_d};
    const Interval<Rational> FloatConstants<Rational>::pi = default_irrational_pi();
    const Interval<Rational> FloatConstants<Rational>::two_pi = default_irrational_two_pi();
    const Interval<Rational> FloatConstants<Rational>::rec_two_pi = default_irrational_rec_two_pi();

    IRational default_irrational_two_pi() {
        IRational r{FloatConstants<Rational>::pi};
        r.lb_ref() *= 2;
        r.ub_ref() *= 2;
        return r;
    }

    IRational default_irrational_rec_two_pi() {
        const IRational& two_pi = FloatConstants<Rational>::two_pi;
        return IRational{reciprocal(two_pi.ub()), reciprocal(two_pi.lb())};
    }

    IRational rational_rec_two_pi(unsigned precision) {
        DynamicMPFRNumber lb_mpfr_pi(precision);
        DynamicMPFRNumber ub_mpfr_pi(precision);
        mpfr_const_pi(lb_mpfr_pi, MPFR_RNDD);
        mpfr_mul_2ui(lb_mpfr_pi, lb_mpfr_pi, 1, MPFR_RNDD);
        mpfr_const_pi(ub_mpfr_pi, MPFR_RNDU);
        mpfr_mul_2ui(ub_mpfr_pi, ub_mpfr_pi, 1, MPFR_RNDU);
        Rational lb_2pi;
        Rational ub_2pi;
        mpfr_get_q(lb_2pi.get_mpq_t(), lb_mpfr_pi);
        mpfr_get_q(ub_2pi.get_mpq_t(), ub_mpfr_pi);
        invert(lb_2pi);
        invert(ub_2pi);
        return IRational{std::move(ub_2pi),std::move(lb_2pi)};
    }
}
}
