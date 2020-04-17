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
// Created by Phillip Keldenich on 2019-09-23.
//

#pragma once

#include <gmpxx.h>
#include <mpfr.h>

#include <type_traits>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <limits>
#include <climits>
#include <vector>

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cmath>

#include <boost/io/ios_state.hpp>
#include <boost/config.hpp>

#include "ivarp/rounding.hpp"
#include "ivarp/bool.hpp"
#include "ivarp/metaprogramming.hpp"
#include "number/fwd.hpp"
#include "number/device_compat.hpp"
#include "number/exact_less_than.hpp"

namespace ivarp {
    /// A traits template that defines methods for getting lower and
    /// upper bounds as well as possible/definite undefinedness.
    template<typename NT> struct NumberTraits {
        static constexpr bool is_number = false;
    };

    /// A type and an object that can be passed to intervals to set the corresponding bound to infinity;
    /// needed for intervals if the underlying number type has no infinities.
    struct InfinityType {
        constexpr IVARP_HD InfinityType operator-() const noexcept { return InfinityType{}; }
    };

    static IVARP_CUDA_DEVICE_OR_CONSTEXPR InfinityType infinity{};

    template<typename Number> IVARP_HD constexpr static inline std::enable_if_t<!std::is_floating_point<Number>::value, bool>
        is_finite(const Number&) noexcept
    {
        return true;
    }

    template<typename Number> IVARP_HD static inline std::enable_if_t<std::is_floating_point<Number>::value, bool>
        is_finite(Number n) noexcept
    {
        return IVARP_NOCUDA_USE_STD isfinite(n);
    }

    template<typename Number> IVARP_HD static inline bool possibly_undefined(const Number& n) {
        return NumberTraits<Number>::possibly_undefined(n);
    }

    template<typename Number> IVARP_HD static inline bool definitely_defined(const Number& n) {
        return !NumberTraits<Number>::possibly_undefined(n);
    }

    template<typename Number>
        static inline IVARP_HD std::enable_if_t<NumberTraits<Number>::allows_cuda, typename NumberTraits<Number>::BoundType> lb(const Number& n)
    {
        return NumberTraits<Number>::lb(n);
    }

    template<typename Number>
        static inline IVARP_H std::enable_if_t<!NumberTraits<Number>::allows_cuda, typename NumberTraits<Number>::BoundType> lb(const Number& n)
    {
        return NumberTraits<Number>::lb(n);
    }

    template<typename Number>
        static inline IVARP_HD std::enable_if_t<NumberTraits<Number>::allows_cuda, typename NumberTraits<Number>::BoundType> ub(const Number& n)
    {
        return NumberTraits<Number>::ub(n);
    }

    template<typename Number>
        static inline IVARP_H std::enable_if_t<!NumberTraits<Number>::allows_cuda, typename NumberTraits<Number>::BoundType> ub(const Number& n)
    {
        return NumberTraits<Number>::ub(n);
    }

    template<typename IntType> using IsIntegral = std::integral_constant<bool,
        std::is_integral<BareType<IntType>>::value || std::is_same<BareType<IntType>, BigInt>::value
    >;

    template<typename T> struct IsRationalImpl : std::false_type {};
    template<typename A> struct IsRationalImpl<__gmp_expr<mpq_t, A>> : std::true_type {};

    template<typename NumType> using IsNumber = std::integral_constant<bool,
        NumberTraits<BareType<NumType>>::is_number>;

    template<bool IsNumber, typename NumType> struct IsCudaNumberImpl : std::false_type {};
    template<typename NumType> struct IsCudaNumberImpl<true, NumType> : std::integral_constant<bool, NumberTraits<NumType>::allows_cuda> {};
    template<typename NumType> using IsCudaNumber = IsCudaNumberImpl<IsNumber<NumType>::value, NumType>;

    template<typename T> using IsIntOrRational = std::integral_constant<bool,
            IsIntegral<T>::value || IsRational<T>::value>;

    template<typename T, bool B = IsNumber<T>::value> struct IsIntervalType : std::false_type {};
    template<typename T> struct IsIntervalType<T,true> : std::integral_constant<bool, NumberTraits<T>::is_interval> {};

    /// An interval class that also keeps track of definedness issues.
    template<typename NumberType> class Interval;

    // Intervals of float, double and Rational type.
    using IFloat  = Interval<float>;
    using IDouble = Interval<double>;
    using IRational = Interval<Rational>;

    /**
     * @brief Generate a (canonicalized) rational number from one or two integers.
     * Compared to the native GMP interface, this provides two benefits:
     * - It canonicalizes the number; failure to do so can results in difficult-to-diagnose bugs.
     * - It works with _all_ integral types including [unsigned] long long and std::[u]int64_t.
     * There also are versions where one or both of the arguments are BigInt.
     * @return The rational
     */
    template<typename IntegralType1, std::enable_if_t<IsIntegral<IntegralType1>::value,int> = 0>
        static inline Rational rational(IntegralType1 num);
    template<typename IntegralType1, typename IntegralType2>
        static inline std::enable_if_t<IsIntegral<IntegralType1>::value && IsIntegral<IntegralType2>::value, Rational>
            rational(IntegralType1 num, IntegralType2 denom);

    // a metafunction that computes the promoted number type for a set of argument types:
    // computations are done, unless additional promotions to intervals occur due to some operations, using
    // the number type with the highest rank between all arguments.
    // If a promotion to intervals occurs, it uses the interval version of the basic number type.
    // Rank is determined as follows:
    //  - Any interval type has higher rank than any number type. In particular, passing one float interval and
    //    a rational parameter leads to the computation being done on float intervals.
    //  - More precise number types have higher rank.
    template<typename Arg1, typename... Args> struct NumberTypePromotion;
    template<typename... Args> using Promote = typename NumberTypePromotion<BareType<Args>...>::type;
//  template<typename TargetType, typename SourceType> static inline
//      TargetType convert_number(const SourceType& source);
}

#include "number/mpfr.hpp"
#include "number/interval.hpp"
#include "number/traits.hpp"
#include "number/rational.hpp"
#include "number/fixed_point_bounds.hpp"
#include "number/fixed_point_bounds_sqrt.hpp"
#include "number/fixed_point_bounds_sin.hpp"
#include "number/fixed_point_bounds_cos.hpp"
#include "number/float_interval_ops.hpp"
#include "number/interval_comparisons.hpp"
#include "number/bounded_rational.hpp"
#include "number/type_conversions.hpp"
#include "number/minmax.hpp"
#include "number/literals.hpp"
#include "number/factorize.hpp"
