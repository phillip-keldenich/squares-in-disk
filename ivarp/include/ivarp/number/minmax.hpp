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
// Created by Phillip Keldenich on 06.11.19.
//

#pragma once

namespace ivarp {
    template<typename Number> static inline IVARP_HD
        std::enable_if_t<IsCUDANumber<Number>::value && !IsIntervalType<Number>::value, Number>
            minimum(const Number& n1, const Number& n2) noexcept
    {
        return n1 < n2 ? n1 : n2;
    }

    template<typename Number> static inline IVARP_H
        std::enable_if_t<IsNumber<Number>::value && !IsCUDANumber<Number>::value && !IsIntervalType<Number>::value, Number>
            minimum(const Number& n1, const Number& n2)
    {
        return n1 < n2 ? n1 : n2;
    }

    template<typename Number> static inline IVARP_HD
        std::enable_if_t<IsCUDANumber<Number>::value && !IsIntervalType<Number>::value, Number>
            maximum(const Number& n1, const Number& n2) noexcept
    {
        return n2 < n1 ? n1 : n2;
    }

    template<typename Number> static inline IVARP_H
        std::enable_if_t<IsNumber<Number>::value && !IsCUDANumber<Number>::value && !IsIntervalType<Number>::value, Number>
            maximum(const Number& n1, const Number& n2)
    {
        return n2 < n1 ? n1 : n2;
    }

    namespace impl {
        template<typename IntervalType> static inline IVARP_HD
            std::enable_if_t<IsCUDANumber<IntervalType>::value, IntervalType>
                ia_minimum(const IntervalType& i1, const IntervalType& i2) noexcept
        {
            return IntervalType{
                minimum(i1.lb(), i2.lb()), minimum(i1.ub(), i2.ub()),
                i1.possibly_undefined() || i2.possibly_undefined()
            };
        }

        template<typename IntervalType> static inline IVARP_H
            std::enable_if_t<!IsCUDANumber<IntervalType>::value && !IntervalType::has_explicit_infinity, IntervalType>
                ia_minimum(const IntervalType& i1, const IntervalType& i2)
        {
            return IntervalType{
                (std::min)(i1.lb(), i2.lb()), (std::min)(i1.ub(), i2.ub()),
                i1.possibly_undefined() || i2.possibly_undefined()
            };
        }

        template<typename IntervalType> static inline IVARP_HD
            std::enable_if_t<IsCUDANumber<IntervalType>::value, IntervalType>
                ia_maximum(const IntervalType& i1, const IntervalType& i2) noexcept
        {
            return IntervalType{
                maximum(i1.lb(), i2.lb()), maximum(i1.ub(), i2.ub()),
                i1.possibly_undefined() || i2.possibly_undefined()
            };
        }

        template<typename IntervalType> static inline IVARP_H
            std::enable_if_t<!IsCUDANumber<IntervalType>::value && !IntervalType::has_explicit_infinity, IntervalType>
                ia_maximum(const IntervalType& i1, const IntervalType& i2)
        {
            return IntervalType{
                (std::min)(i1.lb(), i2.lb()), (std::min)(i1.ub(), i2.ub()),
                i1.possibly_undefined() || i2.possibly_undefined()
            };
        }

        template<typename IntervalType> static inline IVARP_H
            std::enable_if_t<IntervalType::has_explicit_infinity, IntervalType>
                ia_minimum(const IntervalType& i1, const IntervalType& i2)
        {
            IntervalType result;
            if(i1.finite_lb() && i2.finite_lb()) {
                result.set_lb((std::min)(i1.lb(), i2.lb()));
            } else {
                result.set_lb(-infinity);
            }

            if(i1.finite_ub() && i2.finite_ub()) {
                result.set_ub((std::min)(i1.ub(), i2.ub()));
            } else if(i2.finite_ub()) {
                result.set_ub(i2.ub());
            } else if(i1.finite_ub()) {
                result.set_ub(i1.ub());
            } else {
                result.set_ub(infinity);
            }

            result.set_undefined(i1.possibly_undefined() | i2.possibly_undefined());
            return result;
        }

        template<typename IntervalType> static inline IVARP_H
            std::enable_if_t<IntervalType::has_explicit_infinity, IntervalType>
                ia_maximum(const IntervalType& i1, const IntervalType& i2)
        {
            IntervalType result;
            if(i1.finite_ub() && i2.finite_ub()) {
                result.set_ub((std::max)(i1.ub(), i2.ub()));
            } else {
                result.set_ub(infinity);
            }

            if(i1.finite_lb() && i2.finite_lb()) {
                result.set_lb((std::max)(i1.lb(), i2.lb()));
            } else if(i1.finite_lb()) {
                result.set_lb(i1.lb());
            } else if(i2.finite_lb()) {
                result.set_lb(i2.lb());
            } else {
                result.set_lb(-infinity);
            }

            result.set_undefined(i1.possibly_undefined() | i2.possibly_undefined());
            return result;
        }
    }

    template<typename Number> static inline IVARP_HD
        std::enable_if_t<IsIntervalType<Number>::value && IsCUDANumber<Number>::value, Number>
            minimum(const Number& n1, const Number& n2) noexcept
    {
       return impl::ia_minimum(n1, n2);
    }

    template<typename Number> static inline IVARP_H
        std::enable_if_t<IsIntervalType<Number>::value && !IsCUDANumber<Number>::value, Number>
            minimum(const Number& n1, const Number& n2)
    {
       return impl::ia_minimum(n1, n2);
    }

    template<typename Number> static inline IVARP_HD
        std::enable_if_t<IsIntervalType<Number>::value && IsCUDANumber<Number>::value, Number>
            maximum(const Number& n1, const Number& n2) noexcept
    {
       return impl::ia_maximum(n1, n2);
    }

    template<typename Number> static inline IVARP_H
        std::enable_if_t<IsIntervalType<Number>::value && !IsCUDANumber<Number>::value, Number>
            maximum(const Number& n1, const Number& n2)
    {
       return impl::ia_maximum(n1, n2);
    }
}
