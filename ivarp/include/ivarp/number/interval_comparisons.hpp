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
// Created by Phillip Keldenich on 28.10.19.
//

#pragma once

namespace ivarp {
    /// Implementation of interval comparison operator<; we have three variants for each operator:
    /// Number op Interval, Interval op Number, and Interval op Interval.
    template<typename NumberType, typename IntervalNumberType> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value && AllAllowCuda<NumberType, IntervalNumberType>::value, IBool>
            operator<(const NumberType& n1, const IntervalNumberType& n2) noexcept
    {
        return {
            !n2.possibly_undefined() && n2.below_lb(n1),
            n2.possibly_undefined() || n2.below_ub(n1)
        };
    }

    template<typename NumberType, typename IntervalNumberType> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value && !AllAllowCuda<NumberType, IntervalNumberType>::value, IBool>
            operator<(const NumberType& n1, const IntervalNumberType& n2)
    {
        return {
            !n2.possibly_undefined() && n2.below_lb(n1),
            n2.possibly_undefined() || n2.below_ub(n1)
        };
    }

    template<typename NumberType, typename IntervalNumberType> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value && AllAllowCuda<NumberType, IntervalNumberType>::value, IBool>
            operator<(const IntervalNumberType& n1, const NumberType& n2) noexcept
    {
        return {
            !n1.possibly_undefined() && n1.above_ub(n2),
            n1.possibly_undefined() || n1.above_lb(n2)
        };
    }

    template<typename NumberType, typename IntervalNumberType> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value && !AllAllowCuda<NumberType, IntervalNumberType>::value, IBool>
            operator<(const IntervalNumberType& n1, const NumberType& n2) noexcept
    {
        return {
            !n1.possibly_undefined() && n1.above_ub(n2),
            n1.possibly_undefined() || n1.above_lb(n2)
        };
    }

namespace impl {
    /// Implementation of < when at least one of the intervals has infinity information outside the number type.
    template<typename T1, typename T2, bool ExplicitInfinity1 = T1::has_explicit_infinity,
             bool ExplicitInfinity2 = T2::has_explicit_infinity, bool CUDA = AllAllowCuda<T1,T2>::value>
        struct IntervalLessImpl
    {
        static IBool less(const T1& t1, const T2& t2) noexcept {
            if(t1.possibly_undefined() || t2.possibly_undefined()) {
                return {false, true};
            }

            return {
                t1.finite_ub() && t2.finite_lb() && t1.ub() < t2.lb(),
                !t1.finite_lb() || !t2.finite_ub() || t1.lb() < t2.ub()
            };
        }
    };

    /// Implementation of < for intervals on number types where infinities are implemented in the number type.
    template<typename T1, typename T2> struct IntervalLessImpl<T1,T2,false,false,false> {
        static IBool less(const T1& t1, const T2& t2) {
            return {
                !t1.possibly_undefined() && !t2.possibly_undefined() && t1.ub() < t2.lb(),
                t1.possibly_undefined() || t2.possibly_undefined() || t1.lb() < t2.ub()
            };
        }
    };
    template<typename T1, typename T2> struct IntervalLessImpl<T1,T2,false,false,true> {
        static IVARP_HD IBool less(const T1& t1, const T2& t2) noexcept {
            return {
                !t1.possibly_undefined() && !t2.possibly_undefined() && t1.ub() < t2.lb(),
                t1.possibly_undefined() || t2.possibly_undefined() || t1.lb() < t2.ub()
            };
        }
    };
}

    template<typename IntervalNumberType1, typename IntervalNumberType2> static inline IVARP_HD
        std::enable_if_t<IsIntervalType<IntervalNumberType1>::value &&
                         IsIntervalType<IntervalNumberType2>::value && AllAllowCuda<IntervalNumberType1,IntervalNumberType2>::value, IBool>
            operator<(const IntervalNumberType1& n1, const IntervalNumberType2& n2) noexcept
    {
        return impl::IntervalLessImpl<IntervalNumberType1,IntervalNumberType2>::less(n1, n2);
    }

    template<typename IntervalNumberType1, typename IntervalNumberType2> static inline IVARP_H
        std::enable_if_t<IsIntervalType<IntervalNumberType1>::value &&
                         IsIntervalType<IntervalNumberType2>::value && !AllAllowCuda<IntervalNumberType1,IntervalNumberType2>::value, IBool>
            operator<(const IntervalNumberType1& n1, const IntervalNumberType2& n2) noexcept
    {
        return impl::IntervalLessImpl<IntervalNumberType1,IntervalNumberType2>::less(n1, n2);
    }

    /// Implement the other operators in terms of <.
    template<typename NumberType1, typename NumberType2> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && AllAllowCuda<NumberType1,NumberType2>::value &&
                         (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator>(const NumberType1& a, const NumberType2& b) noexcept
    {
        return b < a;
    }

    template<typename NumberType1, typename NumberType2> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && !AllAllowCuda<NumberType1,NumberType2>::value &&
                         (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator>(const NumberType1& a, const NumberType2& b)
    {
        return b < a;
    }

    template<typename NumberType1, typename NumberType2> static inline IVARP_HD
    std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && AllAllowCuda<NumberType1,NumberType2>::value &&
                     (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator>=(const NumberType1& a, const NumberType2& b) noexcept
    {
        return !(a < b);
    }

    template<typename NumberType1, typename NumberType2> static inline IVARP_H
    std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && !AllAllowCuda<NumberType1,NumberType2>::value &&
                     (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator>=(const NumberType1& a, const NumberType2& b)
    {
        return !(a < b);
    }

    template<typename NumberType1, typename NumberType2> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && AllAllowCuda<NumberType1,NumberType2>::value &&
                     (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator<=(const NumberType1& a, const NumberType2& b) noexcept
    {
        return !(b < a);
    }

    template<typename NumberType1, typename NumberType2> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && !AllAllowCuda<NumberType1,NumberType2>::value &&
                     (IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value), IBool>
            operator<=(const NumberType1& a, const NumberType2& b)
    {
        return !(b < a);
    }

namespace impl {
    /// Implementation of interval overlap test if at least one interval type has infinities outside of the number type.
    template<typename T1, typename T2, bool ExplicitInfinity1 = T1::has_explicit_infinity,
             bool ExplicitInfinity2 = T2::has_explicit_infinity, bool CUDA = AllAllowCuda<T1,T2>::value>
        struct IntervalOverlapImpl
    {
        static bool overlap(const T1& a, const T2& b) noexcept {
            return !(a.finite_ub() && b.finite_lb() && a.ub() < b.lb()) &&
                   !(b.finite_ub() && a.finite_lb() && b.ub() < a.lb());
        }
    };

    template<typename T1, typename T2> struct IntervalOverlapImpl<T1,T2,false,false,false> {
        static bool overlap(const T1& a, const T2& b) {
            return !(a.ub() < b.lb()) && !(b.ub() < a.lb());
        }
    };

    template<typename T1, typename T2> struct IntervalOverlapImpl<T1,T2,false,false,true> {
        static IVARP_HD bool overlap(const T1& a, const T2& b) noexcept {
            return !(a.ub() < b.lb()) && !(b.ub() < a.lb());
        }
    };
}

    /// Check for overlapping intervals (needed for ==).
    template<typename IntervalType1, typename IntervalType2> static inline IVARP_HD
        EnableForCudaNTs<bool, IntervalType1, IntervalType2> overlap(const IntervalType1& a, const IntervalType2& b) noexcept
    {
        return impl::IntervalOverlapImpl<IntervalType1,IntervalType2>::overlap(a,b);
    }

    template<typename IntervalType1, typename IntervalType2> static inline IVARP_H
        DisableForCudaNTs<bool, IntervalType1, IntervalType2> overlap(const IntervalType1& a, const IntervalType2& b)
    {
        return impl::IntervalOverlapImpl<IntervalType1,IntervalType2>::overlap(a,b);
    }

    /// Equality comparison.
    template<typename NumberType, typename IntervalNumberType> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value && AllAllowCuda<NumberType,IntervalNumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value, IBool>
            operator==(const NumberType& a, const IntervalNumberType& b) noexcept
    {
        return {
            !b.possibly_undefined() && b.singleton() && b.lb() == a,
            b.possibly_undefined() || b.contains(a)
        };
    }
    template<typename NumberType, typename IntervalNumberType> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value && !AllAllowCuda<NumberType,IntervalNumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value, IBool>
            operator==(const NumberType& a, const IntervalNumberType& b)
    {
        return {
            !b.possibly_undefined() && b.singleton() && b.lb() == a,
            b.possibly_undefined() || b.contains(a)
        };
    }

    template<typename NumberType, typename IntervalNumberType> static inline IVARP_HD
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value && AllAllowCuda<NumberType,IntervalNumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value, IBool>
            operator==(const IntervalNumberType& a, const NumberType& b) noexcept
    {
        return b == a;
    }
    template<typename NumberType, typename IntervalNumberType> static inline IVARP_H
        std::enable_if_t<IsNumberOrInt<NumberType>::value && !IsIntervalType<NumberType>::value && !AllAllowCuda<NumberType,IntervalNumberType>::value &&
                         IsIntervalType<IntervalNumberType>::value, IBool>
            operator==(const IntervalNumberType& a, const NumberType& b) noexcept
    {
        return b == a;
    }

    template<typename IntervalType1, typename IntervalType2> static inline IVARP_HD
        std::enable_if_t<IsIntervalType<IntervalType1>::value && IsIntervalType<IntervalType2>::value && AllAllowCuda<IntervalType1,IntervalType2>::value, IBool>
            operator==(const IntervalType1& a, const IntervalType2& b) noexcept
    {
        return {
            !a.possibly_undefined() && !b.possibly_undefined() &&
            a.singleton() && b.singleton() && a.lb() == b.lb(),
            a.possibly_undefined() || b.possibly_undefined() || overlap(a,b)
        };
    }

    template<typename IntervalType1, typename IntervalType2> static inline IVARP_H
        std::enable_if_t<IsIntervalType<IntervalType1>::value && IsIntervalType<IntervalType2>::value && !AllAllowCuda<IntervalType1,IntervalType2>::value, IBool>
            operator==(const IntervalType1& a, const IntervalType2& b)
    {
        return {
            !a.possibly_undefined() && !b.possibly_undefined() &&
            a.singleton() && b.singleton() && a.lb() == b.lb(),
            a.possibly_undefined() || b.possibly_undefined() || overlap(a,b)
        };
    }

    /// Implementation of != in terms of ==.
    template<typename NumberType1, typename NumberType2> static inline IVARP_HD
        std::enable_if_t<(IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value) &&
                         IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && AllAllowCuda<NumberType1,NumberType2>::value>
            operator!=(const NumberType1& a, const NumberType2& b) noexcept
    {
        return !(a == b);
    }
    template<typename NumberType1, typename NumberType2> static inline IVARP_H
        std::enable_if_t<(IsIntervalType<NumberType1>::value || IsIntervalType<NumberType2>::value) &&
                         IsNumberOrInt<NumberType1>::value && IsNumberOrInt<NumberType2>::value && !AllAllowCuda<NumberType1,NumberType2>::value>
            operator!=(const NumberType1& a, const NumberType2& b)
    {
        return !(a == b);
    }
}
