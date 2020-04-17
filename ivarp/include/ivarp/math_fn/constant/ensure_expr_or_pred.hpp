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
// Created by Phillip Keldenich on 28.12.19.
//

#include "implicitly_convertible.hpp"

namespace ivarp {
namespace impl {
    template<typename T, bool IsEPC = IsExprPredOrConstant<T>::value, bool IsEOP = IsMathExprOrPred<T>::value>
    struct EnsureExprOrPredImpl {
        static_assert(IsEPC, "Invalid type given to EnsureExprOrPredImpl!");
    };
    template<typename T> struct EnsureExprOrPredImpl<T, true, true> {
        using Type = T;
    };
    template<typename T> struct EnsureExprOrPredImpl<T, true, false> {
        using Type = NumberToConstant<T>;
    };

    template<typename T> struct EnsureExprImpl : EnsureExprOrPredImpl<T> {
        static_assert(!IsMathPred<typename EnsureExprOrPredImpl<T>::Type>::value, "Predicate passed to EnsureExpr!");
    };
    template<typename T> struct EnsurePredImpl : EnsureExprOrPredImpl<T> {
        static_assert(IsMathPred<typename EnsureExprOrPredImpl<T>::Type>::value, "Non-predicate passed to EnsurePred!");
    };
}

    template<typename T> using EnsurePred = typename impl::EnsurePredImpl<BareType<T>>::Type;
    template<typename T> using EnsureExpr = typename impl::EnsureExprImpl<BareType<T>>::Type;
    template<typename T> using EnsureExprOrPred = typename impl::EnsureExprOrPredImpl<BareType<T>>::Type;

    /// Turn expressions that are not MathExpressions into MathExpressions.
    /// Simply forward if given a MathExpression.
    template<typename T, std::enable_if_t<IsMathExpr<T>::value, int> = 0>
        static inline T& ensure_expr(T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathExpr<T>::value, int> = 0>
        static inline const T& ensure_expr(const T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathExpr<T>::value, int> = 0>
        static inline T&& ensure_expr(T&& t) noexcept
    {
        static_assert(!std::is_lvalue_reference<T>::value, "Incorrect overload chosen!");
        return static_cast<T&&>(t);
    }
    template<typename T, std::enable_if_t<!IsMathExpr<T>::value && IsExprOrConstant<T>::value, int> = 0>
        static inline auto ensure_expr(T&& t)
    {
        return NumberToConstant<T>{ivarp::forward<T>(t)};
    }

    /// Turn expressions that are not MathPredicates into MathPredicates.
    /// Simply forward if given a MathPredicate.
    template<typename T, std::enable_if_t<IsMathPred<T>::value, int> = 0>
        static inline T& ensure_pred(T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathPred<T>::value, int> = 0>
        static inline const T& ensure_pred(const T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathPred<T>::value, int> = 0>
        static inline T&& ensure_pred(T&& t) noexcept
    {
        static_assert(!std::is_lvalue_reference<T>::value, "Incorrect overload chosen!");
        return static_cast<T&&>(t);
    }
    template<typename T, std::enable_if_t<!IsMathPred<T>::value && IsPredOrConstant<T>::value, int> = 0>
        static inline auto ensure_pred(T&& t)
    {
        return NumberToConstant<T>{ivarp::forward<T>(t)};
    }

    template<typename T, std::enable_if_t<IsMathExprOrPred<T>::value, int> = 0>
        static inline T& ensure_expr_or_pred(T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathExprOrPred<T>::value, int> = 0>
        static inline const T& ensure_expr_or_pred(const T& t) noexcept
    {
        return t;
    }
    template<typename T, std::enable_if_t<IsMathExprOrPred<T>::value, int> = 0>
        static inline T&& ensure_expr_or_pred(T&& t) noexcept
    {
        static_assert(!std::is_lvalue_reference<T>::value, "Incorrect overload chosen!");
        return static_cast<T&&>(t);
    }
    template<typename T, std::enable_if_t<!IsMathExprOrPred<T>::value &&
                                          (IsExprOrConstant<T>::value || IsPredOrConstant<T>::value), int> = 0>
        static inline auto ensure_expr_or_pred(T&& t)
    {
        return NumberToConstant<T>{ivarp::forward<T>(t)};
    }
}
