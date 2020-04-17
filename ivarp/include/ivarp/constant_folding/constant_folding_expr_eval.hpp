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
// Created by Phillip Keldenich on 19.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename Context, typename ArgArray, typename Base, std::int64_t LB, std::int64_t UB>
        struct EvaluateImpl<Context, ConstantFoldedExpr<Base, LB, UB>, ArgArray>
    {
    private:
        using CalledType = ConstantFoldedExpr<Base,LB,UB>;
        using NumberType = typename Context::NumberType;

        /// A simple empty type that depends on some type W.
        /// Used as tag for metaprogramming via overload selection.
        template<typename W> struct WrapType { using Wrapped = W; };

        using NumberTypeTag = WrapType<NumberType>;

        /// Evaluation for builtin floating-point types.
        template<typename Floating> static std::enable_if_t<std::is_floating_point<Floating>::value, Floating>
            do_eval(const CalledType& c, const ArgArray& a, WrapType<Floating>) noexcept
        {
            using CorrespondingIntervalTag = WrapType<Interval<Floating>>;
            return convert_number<Floating>(do_eval(c, a, CorrespondingIntervalTag{}));
        }

        /// Evaluation for IFloat.
        static IFloat do_eval(const CalledType& c, const ArgArray&, WrapType<IFloat>) noexcept {
            return c.ifloat;
        }

        /// Evaluation for IDouble.
        static IDouble do_eval(const CalledType& c, const ArgArray&, WrapType<IDouble>) noexcept {
            return c.idouble;
        }

        /// Evaluation for rationals.
        template<typename = void> static Rational
            do_eval(const CalledType& c, const ArgArray& a, WrapType<Rational>)
        {
            if(!CalledType::is_exact && Context::irrational_precision > default_irrational_precision) {
                return EvaluateImpl<Context, Base, ArgArray>::eval(c.base, a);
            } else {
                return convert_number<Rational>(c.irational);
            }
        }

        /// Evaluation for rational intervals.
        template<typename = void> static IRational
            do_eval(const CalledType& c, const ArgArray& a, WrapType<IRational>)
        {
            if(!CalledType::is_exact && Context::irrational_precision > default_irrational_precision) {
                return EvaluateImpl<Context, Base, ArgArray>::eval(c.base, a);
            } else {
                return c.irational;
            }
        }

    public:
        static auto eval(const CalledType& c, const ArgArray& a) {
            return do_eval(c, a, NumberTypeTag{});
        }
    };
}
}
