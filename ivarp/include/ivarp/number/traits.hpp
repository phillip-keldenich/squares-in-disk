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
// Created by Phillip Keldenich on 05.10.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename FloatType> struct NumberTraitsFloating {
        using BoundType = FloatType;

        static constexpr bool is_interval = false;
        static constexpr bool is_number = true;
        static constexpr bool allows_cuda = true;

        static IVARP_HD bool possibly_undefined(FloatType f) noexcept {
            return false;
        }

        static IVARP_HD FloatType lb(FloatType f) noexcept {
            return f;
        }

        static IVARP_HD FloatType ub(FloatType f) noexcept {
            return f;
        }
    };
}

    template<> struct NumberTraits<float> : impl::NumberTraitsFloating<float> {};
    template<> struct NumberTraits<double> : impl::NumberTraitsFloating<double> {};
    template<> struct NumberTraits<const float> : impl::NumberTraitsFloating<const float> {};
    template<> struct NumberTraits<const double> : impl::NumberTraitsFloating<const double> {};

    template<> struct NumberTraits<Rational> {
        using BoundType = Rational;

        static constexpr bool is_interval = false;
        static constexpr bool is_number = true;
        static constexpr bool allows_cuda = false;

        static IVARP_HD bool possibly_undefined(const Rational&) noexcept {
            return false;
        }

        static IVARP_H const Rational& lb(const Rational& r) noexcept {
            return r;
        }

        static IVARP_H const Rational& ub(const Rational& r) noexcept {
            return r;
        }
    };

    template<typename ExprType> struct NumberTraits<__gmp_expr<mpq_t,ExprType>> {
        using BoundType = Rational;

        static constexpr bool is_interval = false;
        static constexpr bool is_number = true;
        static constexpr bool allows_cuda = false;

        static IVARP_HD bool possibly_undefined(const Rational&) noexcept {
            return false;
        }

        static IVARP_H Rational lb(const __gmp_expr<mpq_t,ExprType>& r) noexcept {
            return r;
        }

        static IVARP_H Rational ub(const __gmp_expr<mpq_t,ExprType>& r) noexcept {
            return r;
        }
    };

namespace impl {
    template<typename NumType, bool AllowsCUDA> struct NumberTraitsIntvImpl;

    template<typename NumType> struct NumberTraitsIntvImpl<NumType,false> {
    private:
        using IVType = Interval<NumType>;

    public:
        using BoundType = NumType;

        static constexpr bool is_number = true;
        static constexpr bool is_interval = true;
        static constexpr bool allows_cuda = false;

        static IVARP_H bool possibly_undefined(const IVType& i) noexcept {
            return i.possibly_undefined();
        }

        static IVARP_H const NumType &lb(const IVType& i) noexcept {
            return i.lb();
        }

        static IVARP_H const NumType &ub(const IVType& i) noexcept {
            return i.ub();
        }
    };

    template<typename NumType> struct NumberTraitsIntvImpl<NumType,true> {
    private:
        using IVType = Interval<NumType>;

    public:
        using BoundType = NumType;

        static constexpr bool is_number = true;
        static constexpr bool is_interval = true;
        static constexpr bool allows_cuda = true;

        static IVARP_HD bool possibly_undefined(const IVType& i) noexcept {
            return i.possibly_undefined();
        }

        static IVARP_HD NumType lb(const IVType& i) noexcept {
            return i.lb();
        }

        static IVARP_HD NumType ub(const IVType& i) noexcept {
            return i.ub();
        }
    };
}

    template<typename NumType> struct NumberTraits<Interval<NumType>> :
        impl::NumberTraitsIntvImpl<NumType, NumberTraits<NumType>::allows_cuda>
    {};

    template<typename T> using IsNumberOrInt = std::integral_constant<bool,
        IsNumber<BareType<T>>::value || std::is_integral<BareType<T>>::value ||
        std::is_same<BareType<T>, BigInt>::value
    >;

    template<typename NT, typename ResultType=void> using EnableForCUDANT =
        std::enable_if_t<AllowsCUDA<NT>::value, ResultType>;
    template<typename ResultType, typename... NTs> using EnableForCUDANTs =
        std::enable_if_t<AllAllowCUDA<NTs...>::value, ResultType>;
    template<typename NT, typename ResultType=void> using DisableForCUDANT =
        std::enable_if_t<!AllowsCUDA<NT>::value, ResultType>;
    template<typename ResultType, typename... NTs> using DisableForCUDANTs =
        std::enable_if_t<!AllAllowCUDA<NTs...>::value, ResultType>;
}
