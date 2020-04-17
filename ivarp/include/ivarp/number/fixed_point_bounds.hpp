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
// Created by Phillip Keldenich on 19.12.19.
//

#pragma once

namespace ivarp {
namespace fixed_point_bounds {
    /// Changing this should not be done lightly.
    /// Requires changes (at least) to some tests and the literals, and the printing.
    /// Requires changes to division routine if it is increased above 2**32-1.
    /// Must be a power of 10.
    /// For some weird reason, to avoid tons of CUDA warnings, must be a constexpr function, not variable;
    /// most likely because otherwise, there would be no clean way to handle the case where the address is used.
    static IVARP_HD constexpr std::int64_t denom() noexcept { return 1000000; }

    /// The maximum bound value (represents +inf).
    static IVARP_HD constexpr std::int64_t max_bound() { return INT64_C(9223372036854'775807); }

    /// The minimum bound value (represents -inf).
    static IVARP_HD constexpr std::int64_t min_bound() { return -max_bound(); }

    /// The maximum integer that can be represented as fixed-point value.
    static IVARP_HD constexpr std::int64_t fp_maxint() { return INT64_C(9223372036854); }

    /// The minimum integer that can be represented as fixed-point value.
    static IVARP_HD constexpr std::int64_t fp_minint() { return -fp_maxint(); }

    IVARP_HD static constexpr inline bool is_lb(std::int64_t lb) noexcept {
        return lb > min_bound();
    }
    IVARP_HD static constexpr inline bool is_ub(std::int64_t ub) noexcept {
        return ub < max_bound();
    }
    IVARP_HD static constexpr bool nonzero(std::int64_t lb1, std::int64_t ub1) noexcept {
        return lb1 > 0 || ub1 < 0;
    }
    IVARP_HD static constexpr bool nonzero(std::int64_t lb1, std::int64_t ub1, std::int64_t lb2, std::int64_t ub2) noexcept {
        return nonzero(lb1,ub1) && nonzero(lb2,ub2);
    }

    IVARP_HD static constexpr bool is_finite(std::int64_t x) noexcept {
        return x > min_bound() && x < max_bound();
    }

    IVARP_HD static constexpr bool is_finite(std::int64_t l, std::int64_t u) noexcept {
        return l > min_bound() && u < max_bound();
    }

    template<typename BoundsType, typename NT, std::enable_if_t<std::is_floating_point<NT>::value,int> = 0>
        IVARP_HD static inline bool nonnegative(const Interval<NT>& i) noexcept
    {
        if(BoundsType::lb >= 0) {
            return true;
        }
        if(BoundsType::ub <= 0) {
            return false;
        }
        return i.lb() >= 0;
    }

    template<typename BoundsType, typename NT, std::enable_if_t<!std::is_floating_point<NT>::value,int> = 0>
        IVARP_H static inline bool nonnegative(const Interval<NT>& i) noexcept
    {
        if(BoundsType::lb >= 0) {
            return true;
        }
        if(BoundsType::ub <= 0) {
            return false;
        }
        return !i.above_lb(0);
    }

    template<typename BoundsType, typename NT, std::enable_if_t<std::is_floating_point<NT>::value,int> = 0>
        IVARP_HD static inline bool positive(const Interval<NT>& i) noexcept
    {
        if(BoundsType::lb > 0) {
            return true;
        }
        if(BoundsType::ub <= 0) {
            return false;
        }
        return i.lb() > 0;
    }

    template<typename BoundsType, typename NT, std::enable_if_t<!std::is_floating_point<NT>::value,int> = 0>
        IVARP_H static inline bool positive(const Interval<NT>& i) noexcept
    {
        if(BoundsType::lb > 0) {
            return true;
        }
        if(BoundsType::ub <= 0) {
            return false;
        }
        return i.below_lb(0);
    }

    template<typename BoundsType, typename = void> struct BoundsOrUndef {
        static constexpr std::int64_t lb = min_bound();
        static constexpr std::int64_t ub = max_bound();
    };

    template<typename BoundsType>
        struct BoundsOrUndef<BoundsType, MakeVoid<decltype(BoundsType::lb),decltype(BoundsType::ub)>>
    {
        static constexpr std::int64_t lb = BoundsType::lb;
        static constexpr std::int64_t ub = BoundsType::ub;
    };

    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(typename BoundsType, typename NumberType),
        NumberType,
        static inline bool possibly_undefined(const Interval<NumberType>& nt) {
            using WB = BoundsOrUndef<BoundsType>;
            if(is_lb(WB::lb) || is_ub(WB::ub)) {
                return false;
            }
            return nt.possibly_undefined();
        }
    )

    template<typename BoundsType, typename NT, std::enable_if_t<std::is_floating_point<NT>::value,int> = 0>
        IVARP_HD static inline bool nonpositive(const Interval<NT>& i) noexcept
    {
        if(BoundsType::ub <= 0) {
            return true;
        }
        if(BoundsType::lb >= 0) {
            return false;
        }
        return i.ub() <= 0;
    }

    template<typename BoundsType, typename NT, std::enable_if_t<!std::is_floating_point<NT>::value,int> = 0>
        IVARP_H static inline bool nonpositive(const Interval<NT>& i) noexcept
    {
        if(BoundsType::ub <= 0) {
            return true;
        }
        if(BoundsType::lb >= 0) {
            return false;
        }
        return !i.below_ub(0);
    }

    template<typename BoundsType, typename NT, std::enable_if_t<std::is_floating_point<NT>::value,int> = 0>
        IVARP_HD static inline bool negative(const Interval<NT>& i) noexcept
    {
        if(BoundsType::ub < 0) {
            return true;
        }
        if(BoundsType::lb >= 0) {
            return false;
        }
        return i.ub() < 0;
    }

    template<typename BoundsType, typename NT, std::enable_if_t<!std::is_floating_point<NT>::value,int> = 0>
        IVARP_H static inline bool negative(const Interval<NT>& i) noexcept
    {
        if(BoundsType::ub < 0) {
            return true;
        }
        if(BoundsType::lb >= 0) {
            return false;
        }
        return i.above_ub(0);
    }

    struct UnboundedPredicate {
        static constexpr bool lb = false;
        static constexpr bool ub = true;
    };

    struct Unbounded {
        static constexpr std::int64_t lb = min_bound();
        static constexpr std::int64_t ub = max_bound();
    };

    enum class Order {
        LT, LE, EQ, GE, GT, UNKNOWN
    };

    static constexpr Order iv_order(std::int64_t lba, std::int64_t uba,
                                    std::int64_t lbb, std::int64_t ubb) noexcept
    {
        if(is_ub(uba) && is_lb(lbb)) {
            if (uba < lbb) {
                return Order::LT;
            }
            if (uba == lbb) {
                if (is_ub(ubb) && is_lb(lba) && ubb == lba) {
                    return Order::EQ;
                }
                return Order::LE;
            }
        }

        if (is_lb(lba) && is_ub(ubb)) {
            if (ubb < lba) {
                return Order::GT;
            }
            if (ubb == lba) {
                return Order::GE;
            }
        }

        return Order::UNKNOWN;
    }
}
}

#include "fixed_point_bounds_impl.hpp"

namespace ivarp {
namespace fixed_point_bounds {
    /// Compute a + b in fixed-point arithmetic, rounding down.
    IVARP_HD static inline constexpr std::int64_t fp_add_rd(std::int64_t a, std::int64_t b) noexcept {
        return (a <= min_bound() || b <= min_bound()) ? min_bound() : fimpl::fp_add(a, b);
    }

    /// Compute a + b in fixed-point arithmetic, rounding up.
    IVARP_HD static inline constexpr std::int64_t fp_add_ru(std::int64_t a, std::int64_t b) noexcept {
        return (a >= max_bound() || b >= max_bound()) ? max_bound() : fimpl::fp_add(a, b);
    }

    /// Compare fixed-point < int.
    IVARP_HD static inline constexpr bool fp_less_than_i(std::int64_t fpval, std::int64_t ival) noexcept {
        return fpval / denom() < ival;
    }

    /// Compare int < fixed-point.
    IVARP_HD static inline constexpr bool i_less_than_fp(std::int64_t ival, std::int64_t fpval) noexcept {
        return ival < fpval / denom() || (ival == fpval / denom() && fpval > 0 && fpval % denom() != 0);
    }

    /// Transform int to fixed-point.
    IVARP_HD static inline constexpr std::int64_t int_to_fp(std::int64_t intval) noexcept {
        return (intval < 0) ? -fimpl::saturated_umul(-intval, denom()) : fimpl::saturated_umul(intval, denom());
    }

    IVARP_H static inline Rational fp_to_rational(std::int64_t fpval) {
        return rational(fpval, denom());
    }

    IVARP_H static inline IRational fp_to_rational_interval(std::int64_t lb, std::int64_t ub) {
        bool def_defined = is_lb(lb) || is_ub(ub);
        IRational result{fp_to_rational(lb), fp_to_rational(ub), !def_defined};
        if(!is_lb(lb)) {
            result.set_lb(-infinity);
        }
        if(!is_ub(ub)) {
            result.set_ub(infinity);
        }
        return result;
    }

    template<typename TargetType> static inline IVARP_H
        Interval<TargetType> fp_to_interval(std::int64_t lb, std::int64_t ub)
    {
        return convert_number<Interval<TargetType>>(fp_to_rational_interval(lb,ub));
    }

    /// Compute a * b in fixed-point arithmetic, rounding down.
    IVARP_HD static inline constexpr std::int64_t fp_mul_rd(std::int64_t a, std::int64_t b) noexcept {
        return (a < 0) ^ (b < 0) ? -fimpl::fp_do_mul_ru((ivarp::abs)(a), (ivarp::abs)(b)) :
            fimpl::fp_do_mul_rd((ivarp::abs)(a), (ivarp::abs)(b));
    }

    /// Compute a * b in fixed-point arithmetic, rounding up.
    IVARP_HD static inline constexpr std::int64_t fp_mul_ru(std::int64_t a, std::int64_t b) noexcept {
        return (a < 0) ^ (b < 0) ? -fimpl::fp_do_mul_rd((ivarp::abs)(a), (ivarp::abs)(b)) :
            fimpl::fp_do_mul_ru((ivarp::abs)(a), (ivarp::abs)(b));
    }

    /// Do fixed-point division, rounding down.
    IVARP_HD static inline constexpr std::int64_t fp_div_rd(std::int64_t a, std::int64_t b) noexcept {
        return (a < 0) ^ (b < 0) ? -fimpl::fp_do_div_ru((ivarp::abs)(a), (ivarp::abs)(b)) :
            fimpl::fp_do_div_rd((ivarp::abs)(a), (ivarp::abs)(b));
    }

    /// Do fixed-point division, rounding up.
    IVARP_HD static inline constexpr std::int64_t fp_div_ru(std::int64_t a, std::int64_t b) noexcept {
        return (a < 0) ^ (b < 0) ? -fimpl::fp_do_div_rd((ivarp::abs)(a), (ivarp::abs)(b)) :
            fimpl::fp_do_div_ru((ivarp::abs)(a), (ivarp::abs)(b));
    }

    /// Compute the bounds of the result of a fixed-point interval addition.
    IVARP_HD static inline constexpr std::int64_t fp_iv_add_lb(std::int64_t lb_a, std::int64_t lb_b) noexcept {
        return fp_add_rd(lb_a, lb_b);
    }
    IVARP_HD static inline constexpr std::int64_t fp_iv_add_ub(std::int64_t ub_a, std::int64_t ub_b) noexcept {
        return fp_add_ru(ub_a, ub_b);
    }

    /// Compute the bounds of the result of a fixed-point interval multiplication.
    IVARP_HD static inline constexpr std::int64_t fp_iv_mul_lb(std::int64_t lb_a, std::int64_t ub_a,
                                                               std::int64_t lb_b, std::int64_t ub_b) noexcept
    {
        if((!is_lb(lb_a) && ub_b > 0) || (!is_lb(lb_b) && ub_a > 0)) {
            return min_bound();
        }
        if((!is_ub(ub_a) && lb_b < 0) || (!is_ub(ub_b) && lb_a < 0)) {
            return min_bound();
        }

        std::int64_t r1 = fp_mul_rd(lb_a, lb_b);
        std::int64_t r2 = fp_mul_rd(lb_a, ub_b);
        std::int64_t r3 = fp_mul_rd(ub_a, lb_b);
        std::int64_t r4 = fp_mul_rd(ub_a, ub_b);
        return (ivarp::min)((ivarp::min)(r1, r2), (ivarp::min)(r3, r4));
    }
    IVARP_HD static inline constexpr std::int64_t fp_iv_mul_ub(std::int64_t lb_a, std::int64_t ub_a,
                                                               std::int64_t lb_b, std::int64_t ub_b) noexcept
    {
        if((!is_ub(ub_a) && ub_b > 0) || (!is_ub(ub_b) && ub_a > 0)) {
            return max_bound();
        }
        if((!is_lb(lb_a) && lb_b < 0) || (!is_lb(lb_b) && lb_a < 0)) {
            return max_bound();
        }

        std::int64_t r1 = fp_mul_ru(lb_a, lb_b);
        std::int64_t r2 = fp_mul_ru(lb_a, ub_b);
        std::int64_t r3 = fp_mul_ru(ub_a, lb_b);
        std::int64_t r4 = fp_mul_ru(ub_a, ub_b);
        return (ivarp::max)((ivarp::max)(r1, r2), (ivarp::max)(r3, r4));
    }

    IVARP_HD static inline constexpr std::int64_t fp_iv_div_lb(std::int64_t lb_a, std::int64_t ub_a,
                                                               std::int64_t lb_b, std::int64_t ub_b) noexcept
    {
        if((lb_b < 0 && ub_b > 0) || (lb_b == 0 && ub_b == 0)) {
            return min_bound();
        }
        if(lb_a == 0 && ub_a == 0) {
            return 0;
        }
        if(lb_b == 0) {
            if(lb_a < 0) {
                return min_bound();
            } else {
                return fp_div_rd(lb_a, ub_b);
            }
        }
        if(ub_b == 0) {
            if(ub_a > 0) {
                return min_bound();
            } else {
                return fimpl::fp_do_div_rd(-ub_a, -lb_b);
            }
        }
        if(lb_b > 0) {
            // b is positive
            return fp_div_rd(lb_a, lb_a < 0 ? lb_b : ub_b);
        } else {
            // b is negative
            return fp_div_rd(ub_a, ub_a > 0 ? ub_b : lb_b);
        }
    }

    IVARP_HD static inline constexpr std::int64_t minimum(std::int64_t i1) noexcept { return i1; }
    template<typename... I64S>
    IVARP_HD static inline constexpr std::int64_t minimum(std::int64_t i1, std::int64_t i2, I64S... is) noexcept
    {
        std::int64_t omin = minimum(i2, is...);
        return i1 < omin ? i1 : omin;
    }

    IVARP_HD static inline constexpr std::int64_t maximum(std::int64_t i1) noexcept { return i1; }
    template<typename... I64S>
    IVARP_HD static inline constexpr std::int64_t maximum(std::int64_t i1, std::int64_t i2, I64S... is) noexcept
    {
        std::int64_t omin = maximum(i2, is...);
        return i1 > omin ? i1 : omin;
    }

    IVARP_HD static inline constexpr std::int64_t fp_iv_div_ub(std::int64_t lb_a, std::int64_t ub_a,
                                                               std::int64_t lb_b, std::int64_t ub_b) noexcept
    {
        if((lb_b < 0 && ub_b > 0) || (lb_b == 0 && ub_b == 0)) {
            return max_bound();
        }
        if(lb_a == 0 && ub_a == 0) {
            return 0;
        }
        if(lb_b == 0) {
            if(ub_a > 0) {
                return max_bound();
            } else {
                return fp_div_ru(ub_a, ub_b);
            }
        }
        if(ub_b == 0) {
            if(lb_a < 0) {
                return max_bound();
            } else {
                return -fimpl::fp_do_div_ru(lb_a, -lb_b);
            }
        }
        if(lb_b > 0) {
            return fp_div_ru(ub_a, ub_a < 0 ? ub_b : lb_b);
        } else {
            return fp_div_ru(lb_a, lb_a > 0 ? lb_b : ub_b);
        }
    }

    IVARP_HD static inline constexpr std::int64_t fpow_lb(std::int64_t lb, std::int64_t ub, unsigned power) {
        if(!is_lb(lb) && !is_ub(ub)) {
            return min_bound();
        }

        if(power == 0) {
            return int_to_fp(1);
        }

        if(power % 2 == 1 && !is_lb(lb)) {
            return min_bound();
        }

        int result_sign = (power % 2 == 0) ? 1 :
                          (lb < 0) ? -1 : 1;
        std::int64_t op = (power % 2 == 0 && ub < 0) ? ub :
                          (power % 2 == 0 && lb < 0 && ub > 0) ? 0 : lb;
        op = (op < 0) ? -op : op;

        std::int64_t result = op;
        for(unsigned i = 1; i < power; ++i) {
            result = result_sign > 0 ? fp_mul_rd(result, op) : fp_mul_ru(result, op);
        }
        return result * result_sign;
    }

    IVARP_HD static inline constexpr std::int64_t fpow_ub(std::int64_t lb, std::int64_t ub, unsigned power) {
        if(!is_lb(lb) && !is_ub(ub)) {
            return max_bound();
        }

        if(power == 0) {
            return int_to_fp(1);
        }

        int result_sign = (power % 2 == 0) ? 1 :
                          (ub < 0) ? -1 : 1;
        std::int64_t op = (power % 2 == 0) ? ivarp::max(-lb, ub) : ub;
        op = (op < 0) ? -op : op;

        std::int64_t result = op;
        for(unsigned i = 1; i < power; ++i) {
            result = result_sign > 0 ? fp_mul_ru(result, op) : fp_mul_rd(result, op);
        }
        return result * result_sign;
    }

    template<typename IntType, bool IsSigned = std::is_signed<IntType>::value> struct BoundsFromIntType;
    
    template<typename IntType> struct BoundsFromIntType<IntType, false> {
    private:
        static constexpr std::uintmax_t mval = std::numeric_limits<IntType>::max();
        static constexpr std::uintmax_t mallow = std::uintmax_t(fp_maxint());
    
    public:
        static constexpr std::int64_t lb = 0;
        static constexpr std::int64_t ub = (mval <= mallow) ? int_to_fp(mval) : max_bound();
    };
    
    template<typename IntType> struct BoundsFromIntType<IntType, true> {
    private:
        static constexpr std::intmax_t minval = std::numeric_limits<IntType>::min();
        static constexpr std::intmax_t minallow = -fp_maxint();
        static constexpr std::intmax_t maxval = std::numeric_limits<IntType>::max();
        static constexpr std::intmax_t maxallow = fp_maxint();

    public:
        static constexpr std::int64_t lb = (minallow <= minval) ? int_to_fp(minval) : min_bound();
        static constexpr std::int64_t ub = (maxval <= maxallow) ? int_to_fp(maxval) : max_bound();
    };

    class PrintFixedPoint {
    public:
        explicit PrintFixedPoint(std::int64_t x) :
            value(x)
        {}

        std::int64_t value;
    };

    IVARP_EXPORTED_SYMBOL std::ostream& operator<<(std::ostream& out, PrintFixedPoint p);
}
}
