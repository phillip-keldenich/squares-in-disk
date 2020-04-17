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
    /// Convert an integer to a constant; use bounds of the type if possible.
    template<typename IntType, typename Enabler = void> struct IntegralToConstantImpl {
    private:
        using BoundsType = fixed_point_bounds::BoundsFromIntType<IntType>;

    public:
        using Type = MathConstant<IntType, BoundsType::lb, BoundsType::ub>;
    };

    template<typename NumberType> struct NumberToConstantImpl {
    private:
        /// Handle boolean constants.
        template<typename NT, std::enable_if_t<IsBoolean<NT>::value, int> = 0>
            static MathBoolConstant<NT, false, true> type_impl(const NT*);

        /// Handle builtin integral values.
        template<typename NT, std::enable_if_t<std::is_integral<NT>::value && !IsBoolean<NT>::value, int> = 0>
            static typename IntegralToConstantImpl<NT>::Type type_impl(const NT*);

        /// Handle bounded numbers.
        template<typename QT, std::int64_t LB, std::int64_t UB, bool DefDefined>
            static MathConstant<QT, (DefDefined ? LB : fixed_point_bounds::min_bound()),
                                (DefDefined ? UB : fixed_point_bounds::max_bound())>
                type_impl(const fixed_point_bounds::BoundedQ<QT,LB,UB,DefDefined>*);

        /// Handle all other numbers.
        static MathConstant<NumberType, fixed_point_bounds::min_bound(), fixed_point_bounds::max_bound()> type_impl(...);

    public:
        using Type = decltype(type_impl(static_cast<const NumberType*>(nullptr)));
    };
}

    template<typename T> using NumberToConstant = typename impl::NumberToConstantImpl<BareType<T>>::Type;
}
