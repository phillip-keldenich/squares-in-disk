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
// Created by Phillip Keldenich on 2019-10-01.
//

#pragma once

namespace ivarp {
    struct MathOperatorTagUnaryMinus {
        static constexpr PrintOp print_operator = PrintOp::UNARY_MINUS;

        struct EvalBounds {
            template<typename B1> struct Eval {
                static constexpr std::int64_t lb = -B1::ub;
                static constexpr std::int64_t ub = -B1::lb;
            };
        };

        template<typename Context, typename NumberType>
            static IVARP_HD inline EnableForCUDANT<NumberType,NumberType> eval(const NumberType& n) noexcept
        {
            return -n;
        }

        template<typename Context, typename NumberType>
            static IVARP_H inline DisableForCUDANT<NumberType,NumberType> eval(const NumberType& n) noexcept
        {
            return -n;
        }
    };

    template<typename A> using MathMinus = MathUnary<MathOperatorTagUnaryMinus, A>;
    template<typename MathExpr> struct Negate {
        using Type = MathMinus<MathExpr>;

        static inline Type negate(MathExpr&& arg) noexcept {
            return Type{ivarp::move(arg)};
        }
        static inline Type negate(const MathExpr& arg) {
            return Type{arg};
        }
    };
    template<typename Arg> struct Negate<MathUnary<MathOperatorTagUnaryMinus, Arg>> {
        using OldType = MathUnary<MathOperatorTagUnaryMinus, Arg>;
        using Type = Arg;

        static inline Type negate(OldType&& arg) noexcept {
            return ivarp::move(arg.arg);
        }
        static inline Type negate(const OldType& arg) {
            return arg.arg;
        }
    };

    template<typename A, std::enable_if_t<IsMathExpr<A>::value,int> = 0> static inline auto operator-(A&& a) {
        return Negate<BareType<A>>::negate(ivarp::forward<A>(a));
    }
}
