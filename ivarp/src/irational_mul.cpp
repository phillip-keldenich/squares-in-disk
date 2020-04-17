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
// Created by Phillip Keldenich on 15.10.19.
//

#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"

namespace ivarp {
    namespace impl {
        static inline std::tuple<Rational,Rational,bool,bool>
            ia_rational_do_mul(Rational lb1, bool lb1i, const Rational& lb2, bool lb2i,
                               Rational ub1, bool ub1i, const Rational& ub2, bool ub2i)
        {
            bool lb_have_zero = (lb1 == 0 && !lb1i) || (lb2 == 0 && !lb2i);
            bool lbinf = !lb_have_zero && (lb1i || lb2i);
            lb1 *= lb2;

            bool ub_have_zero = (ub1 == 0 && !ub1i);
            bool ubinf = !ub_have_zero && (ub1i || ub2i);
            ub1 *= ub2;
            return {std::move(lb1), std::move(ub1), lbinf, ubinf};
        }
    }

    std::tuple<Rational,Rational,bool,bool> impl::ia_rational_mul(const IRational& r1, const IRational& r2) {
        // f1 positive
        if (r1.lb() >= 0 && r1.finite_lb()) {
            if (r2.lb() >= 0 && r2.finite_lb()) {
                // both positive
                return ia_rational_do_mul(r1.lb(), !r1.finite_lb(), r2.lb(), !r2.finite_lb(),
                                          r1.ub(), !r1.finite_ub(), r2.ub(), !r2.finite_ub());
            } else if (r2.ub() <= 0 && r2.finite_ub()) {
                // f1 positive, f2 negative
                return ia_rational_do_mul(r1.ub(), !r1.finite_ub(), r2.lb(), !r2.finite_lb(),
                                          r1.lb(), !r1.finite_lb(), r2.ub(), !r2.finite_ub());
            } else {
                // f1 positive, f2 mixed
                return ia_rational_do_mul(r1.ub(), !r1.finite_ub(), r2.lb(), !r2.finite_lb(),
                                          r1.ub(), !r1.finite_ub(), r2.ub(), !r2.finite_ub());
            }
        }

        // f1 negative
        if (r1.ub() <= 0 && r1.finite_ub()) {
            if (r2.lb() >= 0 && r2.finite_lb()) {
                // f1 negative, f2 positive
                return ia_rational_do_mul(r1.lb(), !r1.finite_lb(), r2.ub(), !r2.finite_ub(),
                                          r1.ub(), !r1.finite_ub(), r2.lb(), !r2.finite_lb());
            } else if (r2.ub() <= 0 && r2.finite_ub()) {
                // both negative
                return ia_rational_do_mul(r1.ub(), !r1.finite_ub(), r2.ub(), !r2.finite_ub(),
                                          r1.lb(), !r1.finite_lb(), r2.lb(), !r2.finite_lb());
            } else {
                // f1 negative, f2 mixed
                return ia_rational_do_mul(r1.lb(), !r1.finite_lb(), r2.ub(), !r2.finite_ub(),
                                          r1.lb(), !r1.finite_lb(), r2.lb(), !r2.finite_lb());
            }
        }

        // f1 mixed
        if (r2.lb() >= 0 && r2.finite_lb()) {
            // f2 positive
            return ia_rational_do_mul(r1.lb(), !r1.finite_lb(), r2.ub(), !r2.finite_ub(),
                                      r1.ub(), !r1.finite_ub(), r2.ub(), !r2.finite_ub());
        } else if (r2.ub() <= 0 && r2.finite_ub()) {
            // f2 negative
            return ia_rational_do_mul(r1.ub(), !r1.finite_ub(), r2.lb(), !r2.finite_lb(),
                                      r1.lb(), !r1.finite_lb(), r2.lb(), !r2.finite_lb());
        }

        // both mixed
        std::tuple<Rational,Rational,bool,bool> res1 =
                ia_rational_do_mul(r1.lb(), !r1.finite_lb(), r2.ub(), !r2.finite_ub(),
                                   r1.lb(), !r1.finite_lb(), r2.lb(), !r2.finite_lb());
        std::tuple<Rational,Rational,bool,bool> res2 =
                ia_rational_do_mul(r2.lb(), !r2.finite_lb(), r1.ub(), !r1.finite_ub(),
                                   r1.ub(), !r1.finite_ub(), r2.ub(), !r2.finite_ub());
        return {
                (std::min)(std::get<0>(res1), std::get<0>(res2)),
                (std::max)(std::get<1>(res1), std::get<1>(res2)),
                std::get<2>(res1) | std::get<2>(res2),
                std::get<3>(res1) | std::get<3>(res2)
        };
    }
}

