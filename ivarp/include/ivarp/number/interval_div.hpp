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
    template<typename T> IVARP_HD static inline int sgn(const T& val) noexcept {
        return (val > 0) - (val < 0);
    }

    IVARP_H static inline Rational div_rd(const Rational& num, const Rational& den) {
        return num / den;
    }

    IVARP_H static inline Rational div_ru(const Rational& num, const Rational& den) {
        return num / den;
    }

    IVARP_H static inline std::pair<Rational,Rational> ia_div(const Rational& lbnum, const Rational& lbden,
                                                              const Rational& ubnum, const Rational& ubden)
    {
        return {lbnum / lbden, ubnum / ubden};
    }

    // compute the lower bound of an interval division result lbnum/lbden, possibly involving infinities.
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static inline void ia_div_lb(Interval<NumberType>& result, const WithInfSign<NumberType>& lbnum,
                                     const WithInfSign<NumberType>& lbden) noexcept(AllowsCuda<NumberType>::value)
        {
            if(!lbnum.inf_sign && !lbden.inf_sign) {
                // the lower bound does not involve infinities.
                result.set_lb(div_rd(lbnum.number, lbden.number));
            } else if(lbden.inf_sign != 0) {
                // the denominator is (+-)inf
                result.set_lb(0);
            } else {
                result.set_lb(-infinity);
            }
        }
    )

    // compute the upper bound of an interval division result ubnum/ubden, possibly involving infinities.
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static inline void ia_div_ub(Interval<NumberType>& result, const WithInfSign<NumberType>& ubnum,
                                     const WithInfSign<NumberType>& ubden) noexcept(AllowsCuda<NumberType>::value)
        {
            if(!ubnum.inf_sign && !ubden.inf_sign) {
                // the upper bound does not involve infinities.
                result.set_ub(div_ru(ubnum.number, ubden.number));
            } else if(ubden.inf_sign != 0) {
                // denominator is (+-)inf
                result.set_ub(0);
            } else {
                result.set_ub(infinity);
            }
        }
    )

    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static inline Interval<NumberType>
            ia_div(const WithInfSign<NumberType>& lbnum, const WithInfSign<NumberType>& lbden,
                   const WithInfSign<NumberType>& ubnum, const WithInfSign<NumberType>& ubden)
                noexcept(AllowsCuda<NumberType>::value)
        {
            Interval<NumberType> result;
            ia_div_lb(result, lbnum, lbden);
            ia_div_ub(result, ubnum, ubden);
            return result;
        }
    )

    template<typename NumberType> IVARP_HD static inline
        std::enable_if_t<std::is_floating_point<NumberType>::value>
            signs(const Interval<NumberType>& n, int& out_lb, int& out_ub) noexcept
    {
        out_lb = sgn(n.lb());
        out_ub = sgn(n.ub());
    }

    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static inline void
            infinity_signs(const Interval<NumberType>& i, int& out_lb, int& out_ub) noexcept
        {
            out_lb = !i.finite_lb() ? -1 : 0;
            out_ub = !i.finite_ub() ? 1 : 0;
        }
    )

    // handle a potential division by zero (mark as possibly undefined, but also provide a result interval).
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static void divide_by_zero(Interval<NumberType>& num, const Interval<NumberType>& den, int sgn_num_lb,
                                   int sgn_num_ub, int sgn_den_sum) noexcept(AllowsCuda<NumberType>::value)
        {
            num.set_undefined(true);
            if(sgn_den_sum == 0 || sgn_num_ub - sgn_num_lb == 2) {
                // denominator or numerator are truly mixed; nothing better than [-inf,inf]
                num.set_lb(-infinity);
                num.set_ub(infinity);
                return;
            }

            if(sgn_den_sum > 0) {
                // denominator >= 0
                if(sgn_num_lb >= 0) {
                    // positive num
                    num.set_lb(div_rd(num.lb(), den.ub()));
                    num.set_ub(infinity);
                } else {
                    // negative num
                    num.set_ub(div_ru(num.ub(), den.ub()));
                    num.set_lb(-infinity);
                }
            } else {
                // denominator <= 0
                if(sgn_num_lb >= 0) {
                    // positive num
                    num.set_ub(div_ru(num.lb(), den.lb()));
                    num.set_lb(-infinity);
                } else {
                    // negative num
                    num.set_lb(div_rd(num.ub(), den.lb()));
                    num.set_ub(infinity);
                }
            }
        }
    )

    // do interval division where some of the bounds may be infinite (slower than the finite version)
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType), NumberType,
        static inline void divide_by_infinite(Interval<NumberType>& num, const Interval<NumberType>& den)
            noexcept(AllowsCuda<NumberType>::value)
        {
            // combine undefinedness flags
            num.set_undefined(num.possibly_undefined() | den.possibly_undefined());

            // extract signs and infinities
            int sgn_num_lb, sgn_num_ub, sgn_den_lb, sgn_den_ub;
            int isgn_num_lb, isgn_num_ub, isgn_den_lb, isgn_den_ub;
            signs(num, sgn_num_lb, sgn_num_ub);
            signs(den, sgn_den_lb, sgn_den_ub);
            infinity_signs(num, isgn_num_lb, isgn_num_ub);
            infinity_signs(den, isgn_den_lb, isgn_den_ub);

            using PT = WithInfSign<NumberType>;
            const PT num_lb{num.lb(), isgn_num_lb};
            const PT num_ub{num.ub(), isgn_num_ub};
            const PT den_lb{den.lb(), isgn_den_lb};
            const PT den_ub{den.ub(), isgn_den_ub};

            if(sgn_den_lb > 0) {
                // positive denominator
                if(sgn_num_lb >= 0) {
                    // positive numerator
                    num.set_bounds(ia_div(num_lb, den_ub, num_ub, den_lb));
                } else if(sgn_num_ub <= 0) {
                    // negative numerator
                    num.set_bounds(ia_div(num_lb, den_lb, num_ub, den_ub));
                } else {
                    // mixed numerator
                    num.set_bounds(ia_div(num_lb, den_lb, num_ub, den_lb));
                }
            } else if(sgn_den_ub < 0) {
                // negative denominator
                if(sgn_num_lb >= 0) {
                    // positive numerator
                    num.set_bounds(ia_div(num_ub, den_ub, num_lb, den_lb));
                } else if(sgn_num_ub <= 0) {
                    // negative numerator
                    num.set_bounds(ia_div(num_ub, den_lb, num_lb, den_ub));
                } else {
                    // mixed numerator
                    num.set_bounds(ia_div(num_ub, den_ub, num_lb, den_ub));
                }
            } else {
                // possible division by zero
                divide_by_zero(num, den, sgn_num_lb, sgn_num_ub, sgn_den_lb + sgn_den_ub);
            }
        }
    )

    // do an interval division where no bounds are infinite
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename NumberType,typename BNum,typename BDen),
        NumberType,
        static inline void divide_by_finite(Interval<NumberType>& num, const Interval<NumberType>& den)
        {
            using namespace fixed_point_bounds;

            num.set_undefined(num.possibly_undefined() | den.possibly_undefined());
            if(positive<BDen>(den)) {
                // positive denominator
                if(nonnegative<BNum>(num)) {
                    // positive numerator
                    num.set_bounds(ia_div(num.lb(), den.ub(), num.ub(), den.lb()));
                } else if(nonpositive<BNum>(num)) {
                    // negative numerator
                    num.set_bounds(ia_div(num.lb(), den.lb(), num.ub(), den.ub()));
                } else {
                    // mixed numerator
                    num.set_bounds(ia_div(num.lb(), den.lb(), num.ub(), den.lb()));
                }
            } else if(negative<BDen>(den)) {
                // negative denominator
                if(nonnegative<BNum>(num)) {
                    // positive numerator
                    num.set_bounds(ia_div(num.ub(), den.ub(), num.lb(), den.lb()));
                } else if(nonpositive<BNum>(num)) {
                    // negative numerator
                    num.set_bounds(ia_div(num.ub(), den.lb(), num.lb(), den.ub()));
                } else {
                    // mixed numerator
                    num.set_bounds(ia_div(num.ub(), den.ub(), num.lb(), den.ub()));
                }
            } else {
                // possible division by zero
                divide_by_zero(num, den, sgn(num.lb()), sgn(num.ub()), sgn(den.lb()) + sgn(den.ub()));
            }
        }
    )

    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(
            typename NumType,
            typename BNum = fixed_point_bounds::Unbounded,
            typename BDen = fixed_point_bounds::Unbounded
        ), NumType,
        static inline void ia_div(Interval<NumType>& num, const Interval<NumType>& den) {
            using namespace fixed_point_bounds;

            if((is_finite(BNum::lb, BNum::ub) && is_finite(BDen::lb,BDen::ub)) ||
               (is_finite(num) && is_finite(den)))
            {
                divide_by_finite<NumType, BNum, BDen>(num, den);
            } else {
                divide_by_infinite(num, den);
            }
        }
    )
}

    template<typename NumberType, typename Bounds1, typename Bounds2> struct BoundedDiv {
        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline NumberType eval(const NumberType& x1, const NumberType& x2)
                noexcept(std::is_nothrow_copy_constructible<NumberType>::value)
            {
                return x1 / x2;
            }
        )
    };

    template<typename InnerType, typename Bounds1, typename Bounds2>
        struct BoundedDiv<Interval<InnerType>, Bounds1, Bounds2>
    {
        using NumberType = Interval<InnerType>;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline NumberType eval(NumberType x1, const NumberType& x2)
                noexcept(AllowsCuda<InnerType>::value)
            {
                impl::ia_div<InnerType, Bounds1, Bounds2>(x1, x2);
                return x1;
            }
        )
    };
}
