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
// Created by Phillip Keldenich on 20.12.19.
//

#pragma once

namespace ivarp {
namespace literal_impl {
    template<char... C> struct Chars { static constexpr std::size_t length = sizeof...(C); };

    /// Compute the value of the digit sequence as std::int64_t value.
    template<char... Digits> struct IntValueOf;
    template<> struct IntValueOf<> {
        static constexpr std::int64_t multiplier = 1;
        static constexpr std::int64_t value = 0;
    };
    template<char Digit> struct IntValueOf<Digit> {
        static constexpr std::int64_t multiplier = 1;
        static constexpr std::int64_t value = Digit - '0';
    };
    template<char D1, char D2, char... Digits> struct IntValueOf<D1,D2,Digits...> {
        using NextType = IntValueOf<D2,Digits...>;
        static constexpr std::size_t multiplier = 10 * NextType::multiplier;
        static constexpr std::size_t value = multiplier * (D1 - '0') + NextType::value;
    };

    /// Check whether the digit sequence fits into a std::int64_t (the check is not exact; actually stops at
    /// 9000000000000000000 < 9223372036854775807 = 2**63 - 1; this does not matter because the fixed-point
    /// bounds cannot represent these values anyways.)
    template<typename CharsType> struct FitsInt;
    template<> struct FitsInt<Chars<>> {
        static constexpr bool value = true;
    };
    template<char C1, char... Cs> struct FitsInt<Chars<C1, Cs...>> {
        static constexpr bool value = (sizeof...(Cs) < 18 || (sizeof...(Cs) == 18 && C1 < '9'));
    };
    template<typename CharsType, bool Fits = FitsInt<CharsType>::value> struct IntBounds;
    template<typename CharsType> struct IntBounds<CharsType,false> {
        static constexpr std::int64_t lb = INT64_C(8999999999999999999);
        static constexpr std::int64_t ub = std::numeric_limits<std::int64_t>::max();
    };
    template<char... Cs> struct IntBounds<Chars<Cs...>,true> {
        static constexpr std::int64_t lb = IntValueOf<Cs...>::value;
        static constexpr std::int64_t ub = lb;
    };

    template<bool Fits, char... Digits> struct IntLiteralImpl;
    template<char... Digits> struct IntLiteralImpl<true, Digits...> {
        constexpr static auto value() noexcept {
            constexpr std::int64_t v = IntValueOf<Digits...>::value;
            constexpr std::int64_t fpv = fixed_point_bounds::int_to_fp(v);
            return BoundedRational<fpv,fpv>{rational(v)};
        }
    };
    template<char... Digits> struct IntLiteralImpl<false, Digits...> {
        static auto value() noexcept {
            static const char buffer[] = {Digits..., '\0'};
            return BoundedRational<fixed_point_bounds::max_bound(), fixed_point_bounds::max_bound()>{
                Rational(buffer, 10)
            };
        }
    };

    /// An integer literal that tracks bounds using compile-time fixed-point arithmetic.
    template<char... Digits> struct IntLiteral :
        IntLiteralImpl<FitsInt<Chars<Digits...>>::value, Digits...>
    {};
    /// Ignore leading zeros.
    template<char... Digits> struct IntLiteral<'0', Digits...> : IntLiteral<Digits...> {};
}


}
