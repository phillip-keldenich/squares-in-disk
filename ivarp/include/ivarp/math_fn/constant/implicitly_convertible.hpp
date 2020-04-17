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
// Created by Phillip Keldenich on 21.12.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename NumType> struct ImplicitConstantPromotion {
    private:
        /// Handle floating-point values.
        template<typename NT, std::enable_if_t<std::is_floating_point<NT>::value, int> = 0>
            static constexpr std::false_type value_impl(const NT*);

        /// Handle integers, rationals and booleans.
        template<typename NT,
                 std::enable_if_t<IsRational<NT>::value || IsBoolean<NT>::value || IsIntegral<NT>::value, int> = 0>
            static constexpr std::true_type value_impl(const NT*);

        /// Handle interval types.
        template<typename NT, std::enable_if_t<IsIntervalType<NT>::value, int> = 0>
            static constexpr std::true_type value_impl(const NT*);

        /// Handle bounded numbers.
        template<typename TT, std::int64_t LB, std::int64_t UB, bool DefDefined>
            static constexpr std::true_type value_impl(const fixed_point_bounds::BoundedQ<TT,LB,UB,DefDefined>*);

        /// Handle all other types.
        static std::false_type value_impl(...);

    public:
        static constexpr bool value = decltype(value_impl(static_cast<const NumType*>(nullptr)))::value;
    };
}
    /// Metafunction that checks whether a numeric type should be
    /// implicitly convertible to a MathConstant.
    ///  * All built-in boolean, integral types and BigInt are implicitly convertible.
    ///    Integral expressions should represent the number or expression written in the code, except for the
    ///    (minor) caveat of integer overflows.
    ///  * To avoid obvious foot-guns due to floating-point literals or expressions not representing
    ///    the number or expression written in the code, built-in floating point types are not implicitly convertible.
    ///  * Rationals are implicitly convertible.
    ///  * Intervals of floating-point or rational type are implicitly convertible.
    ///  * BoundedRational and BoundedIRational are implicitly convertible.
    template<typename NumType> using ImplicitConstantPromotion = impl::ImplicitConstantPromotion<BareType<NumType>>;

    /// Metafunction that checks whether a type is a MathExpression or implicitly convertible to a MathConstant.
    template<typename T> using IsExprOrConstant =
        std::integral_constant<bool, IsMathExpr<T>::value ||
                                     (ImplicitConstantPromotion<T>::value && !IsBoolean<T>::value)>;

    /// Metafunction that checks whether a type is a MathPredicate or implicitly convertible to MathBoolConstant.
    template<typename T> using IsPredOrConstant =
        std::integral_constant<bool, IsMathPred<T>::value || IsBoolean<T>::value>;

    /// Metafunction that checks whether a type is a MathExpression, MathPredicate or implicitly convertible
    /// to a MathConstant or MathBoolConstant.
    template<typename T> using IsExprPredOrConstant =
        std::integral_constant<bool, IsMathExprOrPred<T>::value || ImplicitConstantPromotion<T>::value>;
}
