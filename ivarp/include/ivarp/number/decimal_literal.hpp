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
    template<typename ReadSoFar, char... Digits> struct ReadToDecimalPoint;
    template<char... ReadSoFar, char... Digits> struct ReadToDecimalPoint<Chars<ReadSoFar...>, '.', Digits...> {
        using BeforePoint = Chars<ReadSoFar...>;
        using AfterPoint = Chars<Digits...>;
    };
    template<char... ReadSoFar> struct ReadToDecimalPoint<Chars<ReadSoFar...>> {
        using BeforePoint = Chars<ReadSoFar...>;
        using AfterPoint = Chars<>;
    };
    template<char... ReadSoFar, char D1, char... Digits>
        struct ReadToDecimalPoint<Chars<ReadSoFar...>, D1, Digits...> :
            ReadToDecimalPoint<Chars<ReadSoFar..., D1>, Digits...>
    {
        static_assert(D1 >= '0' && D1 <= '9', "Invalid digit!");
    };

    template<std::int64_t Multiplier, char... AfterDigs> struct AfterDecimalPointImpl;
    template<std::int64_t Multiplier> struct AfterDecimalPointImpl<Multiplier> {
        static constexpr std::int64_t lb = 0;
        static constexpr std::int64_t ub = 0;
    };
    template<char C1, char... AfterDigs> struct AfterDecimalPointImpl<0l, C1, AfterDigs...> {
        static constexpr std::int64_t lb = 0;
        static constexpr std::int64_t ub = (AllOf<C1 == '0', (AfterDigs == '0')...>::value) ? 0 : 1;
    };
    template<std::int64_t Multiplier, char C1, char... Rest> struct AfterDecimalPointImpl<Multiplier, C1, Rest...> {
        using RestType = AfterDecimalPointImpl<Multiplier/10, Rest...>;
        static constexpr std::int64_t lb = Multiplier * std::int64_t(C1 - '0') + RestType::lb;
        static constexpr std::int64_t ub = Multiplier * std::int64_t(C1 - '0') + RestType::ub;
    };

    template<char... AfterDigs> struct AfterDecimalPoint :
        AfterDecimalPointImpl<fixed_point_bounds::denom()/10, AfterDigs...>
    {};

    template<typename B, typename A> struct DecimalBounds;
    template<char... B, char... A> struct DecimalBounds<Chars<B...>, Chars<A...>> {
        static constexpr std::int64_t int_fp_value = ivarp::fixed_point_bounds::int_to_fp(IntValueOf<B...>::value);
        static constexpr std::int64_t lb =
            ivarp::fixed_point_bounds::fimpl::fp_uadd(int_fp_value, AfterDecimalPoint<A...>::lb);
        static constexpr std::int64_t ub =
            ivarp::fixed_point_bounds::fimpl::fp_uadd(int_fp_value, AfterDecimalPoint<A...>::ub);
    };

    template<char... Digits> struct DecimalLiteral {
        using RTP = ReadToDecimalPoint<Chars<>, Digits...>;
        using BP = typename RTP::BeforePoint;
        using AP = typename RTP::AfterPoint;
        using Bounds = DecimalBounds<BP, AP>;

    private:
        template<char... B, char... A> static void fill_buffer(char* buffer, Chars<B...>, Chars<A...>) noexcept {
            const char b[] = {B..., A..., '/', '1'};
            char* after_1 = std::copy(std::begin(b), std::end(b), buffer);
            char* end_0 = after_1 + sizeof...(A);
            for(; after_1 < end_0; ++after_1) {
                *after_1 = '0';
            }
            *after_1 = '\0';
        }

    public:
        static auto value() {
            char buffer[BP::length + 2*AP::length + 3];
            fill_buffer(buffer, BP{}, AP{});
            BoundedRational<Bounds::lb, Bounds::ub> result{Rational(buffer,10)};
            result.value.canonicalize();
            return result;
        }
    };
    template<char... Digits> struct DecimalLiteral<'0', Digits...> : DecimalLiteral<Digits...> {};
}
}
