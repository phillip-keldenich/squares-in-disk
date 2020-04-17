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
// Created by Phillip Keldenich on 19.01.20.
//

#pragma once

#include <type_traits>

/**
 * @file stdcpp.hpp
 *
 * Some utilities carried over from the standard library,
 * but with #IVARP_HD annotations so we can use them in CUDA as well.
 */
namespace ivarp {
    /// Forwarding lvalues as lvalues.
    template<typename T> constexpr static inline IVARP_HD T&& forward(typename std::remove_reference<T>::type& t) noexcept {
        return static_cast<T&&>(t);
    }
    /// Forwarding rvalues as rvalues.
    template<typename T> constexpr static inline IVARP_HD T&& forward(typename std::remove_reference<T>::type&& t) noexcept {
        static_assert(!std::is_lvalue_reference<T>::value,
                      "Template parameter passed to forward does not match lvalueness of argument!");
        return static_cast<T&&>(t);
    }

    /// Forward argument as non-const lvalue.
    template<typename LRValuenessArg, typename T, std::enable_if_t<std::is_lvalue_reference<LRValuenessArg>::value, int> = 0>
        constexpr static inline IVARP_HD T& forward_other(T& v) noexcept
    {
        return v;
    }

    /// Forward argument as const lvalue.
    template<typename LRValuenessArg, typename T, std::enable_if_t<std::is_lvalue_reference<LRValuenessArg>::value, int> = 0>
        constexpr static inline IVARP_HD const T& forward_other(const T& v) noexcept
    {
        return v;
    }

    /// Forward argument as rvalue.
    template<typename LRValuenessArg, typename T, std::enable_if_t<!std::is_lvalue_reference<LRValuenessArg>::value, int> = 0>
        constexpr static inline IVARP_HD std::remove_reference_t<T>&& forward_other(T&& v) noexcept
    {
        static_assert(!std::is_const<LRValuenessArg>::value, "const lvalue reference passed into moving forward_other!");
        return static_cast<std::remove_reference_t<T>&&>(v);
    }

    /// Wrapper around arbitrary types, used to prevent participation in type deduction.
    template<typename T> struct NoTypeDeduction {
        using Type = T;
    };

    /// Maximum and minimum functions.
    template<typename T> constexpr static inline IVARP_HD T min IVARP_PREVENT_MACRO (const T& t1, const typename NoTypeDeduction<T>::Type& t2)
        noexcept(std::is_nothrow_copy_constructible<T>::value && noexcept(t1 < t2))
    {
        return t1 < t2 ? t1 : t2;
    }
    template<typename T> constexpr static inline IVARP_HD T max IVARP_PREVENT_MACRO (const T& t1, const typename NoTypeDeduction<T>::Type& t2)
        noexcept(std::is_nothrow_copy_constructible<T>::value && noexcept(t1 < t2))
    {
        return t1 < t2 ? t2 : t1;
    }

    /// Absolute value.
    template<typename T> constexpr static inline IVARP_HD T abs IVARP_PREVENT_MACRO (const T& t1)
        noexcept(std::is_nothrow_copy_constructible<T>::value && noexcept(-t1 < T(0)))
    {
        return t1 < T(0) ? -t1 : t1;
    }

    /// Move.
    template<typename T> constexpr IVARP_HD std::remove_reference_t<T>&& move(T&& t) noexcept {
        return static_cast<std::remove_reference_t<T>&&>(t);
    }

    // std::swap, has to have different name.
    template<typename T> constexpr IVARP_HD void ivswap(T& v1, T& v2)
        noexcept(std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value)
    {
        T tmp(ivarp::move(v1));
        v1 = ivarp::move(v2);
        v2 = ivarp::move(tmp);
    }
}
