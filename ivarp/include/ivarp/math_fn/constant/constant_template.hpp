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
    /// Representation of a constant expression.
    template<typename A, std::int64_t LB, std::int64_t UB>
        struct MathConstant : MathExpressionBase<MathConstant<A, LB, UB>>
    {
        static_assert(LB <= UB, "Bad lower/upper bound values in MathConstant!");
        static_assert(IsNumber<A>::value || IsIntegral<A>::value,
                      "MathConstant of wrong type!");
        static_assert(std::is_same<A, BareType<A>>::value,
                      "MathConstant of wrong type!");
        static_assert(!IsBoolean<A>::value, "Boolean constant value given to MathConstant!");

        static constexpr bool cuda_supported = false;
        static constexpr std::int64_t lb = LB;
        static constexpr std::int64_t ub = UB;

        using Type = A;

        // do not hide the copy constructor
        template<typename AA, std::enable_if_t<!std::is_same<BareType<AA>, MathConstant>::value, int> = 0>
            explicit IVARP_H MathConstant(AA&& v) noexcept(noexcept(Type(ivarp::forward<AA>(v)))) :
                value(ivarp::forward<AA>(v))
        {
            IVARP_ENSURE_ATLOAD_ROUNDDOWN();
            ifloat = convert_number<IFloat>(value);
            idouble = convert_number<IDouble>(value);
        }

        explicit IVARP_H MathConstant(const fixed_point_bounds::BoundedQ<Type,LB,UB,true>& b)
            noexcept(std::is_nothrow_copy_constructible<A>::value) :
                MathConstant(b.value)
        {}

        explicit IVARP_H MathConstant(fixed_point_bounds::BoundedQ<Type,LB,UB,true>& b)
            noexcept(std::is_nothrow_copy_constructible<A>::value) :
                MathConstant(const_cast<const Type&>(b.value))
        {}

        explicit IVARP_H MathConstant(fixed_point_bounds::BoundedQ<Type,LB,UB,true>&& b)
            noexcept(std::is_nothrow_move_constructible<A>::value) :
                MathConstant(std::move(b.value))
        {}

        IVARP_H MathConstant(const MathConstant& o) noexcept(std::is_nothrow_copy_constructible<A>::value) :
            value(o.value),
            ifloat(o.ifloat),
            idouble(o.idouble)
        {}

        IVARP_H MathConstant(MathConstant&& o) noexcept(std::is_nothrow_move_constructible<A>::value) :
            value(std::move(o.value)),
            ifloat(o.ifloat),
            idouble(o.idouble)
        {}

        IVARP_SUPPRESS_HD
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(typename ResultType), ResultType,
            ResultType as() const noexcept(AllowsCuda<ResultType>::value) {
                return constant_as_impl::AsImpl<MathConstant, ResultType>::as(*this);
            }
        )

        Type value;
        IFloat ifloat;
        IDouble idouble;
    };
}
