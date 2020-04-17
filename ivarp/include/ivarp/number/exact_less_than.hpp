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
// Created by Phillip Keldenich on 28.10.19.
//

#pragma once

/**
 * @file exact_less_than.hpp
 *  Implementation of \f$a < b\f$ or \f$a > b\f$
 *  that is guaranteed to return the correct result,
 *  for the following situations with builtin number types:
 *  \f$a\f$ integral and \f$b\f$ floating-point (or vice-versa),
 *  \f$a\f$ and \f$b\f$ floating-point.
 *  Also works if any involved type is rational.
 *
 *  The problem with the builtin operators is that the comparison is
 *  implemented as if by casting the integer to the floating-point type
 *  before the conversion, which is not exact for large integers and
 *  can thus lead to incorrect results.
 */

namespace ivarp {
namespace impl {
    template<bool GreaterThan, typename IntType> static IVARP_HD inline bool exact_comp_pos(IntType a, double b) {
        constexpr int int_bits = std::numeric_limits<IntType>::digits;
        const double max_exact = IVARP_NOCUDA_USE_STD ldexp(1., std::numeric_limits<double>::digits);
        const double larger_than_max_int = IVARP_NOCUDA_USE_STD ldexp(1., int_bits);

        if(b <= max_exact) {
            return GreaterThan;
        }

        // b is an integral number > max_exact
        if(b >= larger_than_max_int) {
            return !GreaterThan;
        }

        // we can exactly convert b to our integer type
        auto ib = static_cast<IntType>(b);
        return GreaterThan ? a > ib : a < ib;
    }

    template<bool GreaterThan, typename IntType> static IVARP_HD inline bool exact_comp_neg(IntType a, double b) {
        // a is below -max_exact
        constexpr int int_bits = std::numeric_limits<IntType>::digits;

        const double max_exact = IVARP_NOCUDA_USE_STD ldexp(1., std::numeric_limits<double>::digits);
        const double min_int = IVARP_NOCUDA_USE_STD ldexp(-1., int_bits);

        if(b >= -max_exact)  {
            return !GreaterThan;
        }

        // b is integral.
        if(b < min_int) {
            // b is below any integer values
            return GreaterThan;
        }

        // b is integral and in range.
        auto ib = static_cast<IntType>(b);
        return GreaterThan ? a > ib : a < ib;
    }

    template<bool GreaterThan, typename UIntType> static inline IVARP_HD
        std::enable_if_t<std::is_unsigned<UIntType>::value, bool>
            exact_comp(UIntType a, double b) noexcept
    {
        constexpr UIntType max_exact = UIntType(1u) << unsigned(std::numeric_limits<double>::digits);

        // an overflowing int should be rare, therefore a jump is probably good to predict
        if(BOOST_LIKELY(a <= max_exact)) {
            auto da = static_cast<double>(a);
            return GreaterThan ? da > b : da < b;
        }

        return exact_comp_pos<GreaterThan>(a, b);
    }

    template<bool GreaterThan, typename IntType> static inline IVARP_HD
        std::enable_if_t<std::is_signed<IntType>::value, bool>
            exact_comp(IntType a, double b) noexcept
    {
        constexpr IntType max_exact = IntType(1) << unsigned(std::numeric_limits<double>::digits);
        IntType absa = a < 0 ? -a : a; // the only case where this returns negative is exactly representable

        // an overflowing int should be rare
        if(BOOST_LIKELY(absa <= max_exact)) {
            auto da = static_cast<double>(a);
            return GreaterThan ? da > b : da < b;
        }

        if(a < 0) {
            return exact_comp_neg<GreaterThan>(a, b);
        } else {
            return exact_comp_pos<GreaterThan>(a, b);
        }
    }

    /// Handling the case where the int does not necessarily fit into a double's mantissa.
    template<typename Int, typename Float, bool GreaterThan,
             bool FitsFloat = (std::numeric_limits<Int>::digits <= std::numeric_limits<Float>::digits),
             bool FitsDouble = (std::numeric_limits<Int>::digits <= std::numeric_limits<double>::digits)>
    struct ExactCompIntFloat {
        IVARP_HD static bool compare(Int i, Float f) {
            return exact_comp<GreaterThan>(i, f);
        }
    };

    /// Handling the case where the int fits into the float's mantissa.
    template<typename Int, typename Float, bool GreaterThan, bool FD>
        struct ExactCompIntFloat<Int,Float,GreaterThan,true,FD>
    {
        IVARP_HD static bool compare(Int i, Float f) noexcept {
            auto floati = static_cast<Float>(i);
            return GreaterThan ? floati > f : floati < f;
        }
    };

    /// Handling the case where the int fits into a double's mantissa.
    template<typename Int, typename Float, bool GreaterThan>
        struct ExactCompIntFloat<Int,Float,GreaterThan,false,true>
    {
        IVARP_HD static bool compare(Int i, Float f) noexcept {
            auto doublei = static_cast<double>(i);
            double doublef(f);
            return GreaterThan ? doublei > doublef : doublei < doublef;
        }
    };
}

    /// Handling the int/float case.
    template<typename T1, typename T2> static inline IVARP_HD
        std::enable_if_t<std::is_integral<BareType<T1>>::value &&
                         std::is_floating_point<BareType<T2>>::value, bool>
            exact_less_than(T1 a, T2 b) noexcept
    {
        return impl::ExactCompIntFloat<T1, T2, false>::compare(a,b);
    }

    /// Handling the float/int case.
    template<typename T1, typename T2> static inline IVARP_HD
        std::enable_if_t<std::is_integral<BareType<T2>>::value &&
                         std::is_floating_point<BareType<T1>>::value, bool>
            exact_less_than(T1 a, T2 b) noexcept
    {
        return impl::ExactCompIntFloat<T2, T1, true>::compare(b, a);
    }

    /// Easy case: both are already floating point.
    template<typename T1, typename T2> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<BareType<T1>>::value &&
                         std::is_floating_point<BareType<T2>>::value, bool>
            exact_less_than(T1 a, T2 b) noexcept
    {
        return a < b;
    }

    /// Handle the case where at least one is rational and the other is not floating-point.
    template<typename T1, typename T2> static inline IVARP_H
        std::enable_if_t<(IsRational<T1>::value && !std::is_floating_point<T2>::value) || (!std::is_floating_point<T1>::value && IsRational<T2>::value), bool>
            exact_less_than(const T1& a, const T2& b)
    {
        return a < b;
    }

    /// Handle the case where one argument is rational and one is floating-point.
    template<typename T1, typename T2> static inline IVARP_H
        std::enable_if_t<IsRational<T1>::value && std::is_floating_point<T2>::value, bool>
            exact_less_than(const T1& a, T2 b)
    {
        if(b == std::numeric_limits<T2>::infinity()) {
            return true;
        } else if(b == -std::numeric_limits<T2>::infinity()) {
            return false;
        } else {
            return a < b;
        }
    }

    template<typename T1, typename T2> static inline IVARP_H
        std::enable_if_t<IsRational<T2>::value && std::is_floating_point<T1>::value, bool>
            exact_less_than(T1 a, const T2& b)
    {
        if(a == std::numeric_limits<T1>::infinity()) {
            return false;
        } else if(a == -std::numeric_limits<T1>::infinity()) {
            return true;
        } else {
            return a < b;
        }
    }
}
