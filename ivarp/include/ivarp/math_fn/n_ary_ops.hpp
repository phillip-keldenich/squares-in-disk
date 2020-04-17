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
    struct MathNAryMinTag {
        struct EvalBounds {
            template<typename B1, typename... Bs> struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::minimum(B1::lb, Bs::lb...);
                static constexpr std::int64_t ub = fixed_point_bounds::minimum(B1::ub, Bs::ub...);
            };
        };

        static const char* name() noexcept {
            return "min";
        }

        template<typename Context> IVARP_HD static EnableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(typename Context::NumberType a1) noexcept {
            return a1;
        }
        template<typename Context, typename A1> IVARP_H static DisableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(A1&& a1) {
            return ivarp::forward<A1>(a1);
        }

        template<typename Context> static IVARP_HD
            EnableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(typename Context::NumberType a1, typename Context::NumberType a2) noexcept
        {
            return minimum(a1, a2);
        }

        template<typename Context, typename A1, typename A2> static IVARP_H DisableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(A1&& a1, A2&& a2)
        {
            return minimum(ivarp::forward<A1>(a1), ivarp::forward<A2>(a2));
        }

        template<typename Context, typename A1, typename A2, typename A3, typename... Args>
            static IVARP_HD EnableForCudaNT<typename Context::NumberType,typename Context::NumberType> eval(A1&& a1, A2&& a2, A3&& a3, Args&&... args) noexcept
        {
            return eval<Context>(ivarp::forward<A1>(a1), eval<Context>(ivarp::forward<A2>(a2), ivarp::forward<A3>(a3), ivarp::forward<Args>(args)...));
        }

        template<typename Context, typename A1, typename A2, typename A3, typename... Args>
            static IVARP_H DisableForCudaNT<typename Context::NumberType,typename Context::NumberType> eval(A1&& a1, A2&& a2, A3&& a3, Args&&... args)
        {
            return eval<Context>(ivarp::forward<A1>(a1), eval<Context>(ivarp::forward<A2>(a2), ivarp::forward<A3>(a3), ivarp::forward<Args>(args)...));
        }
    };

    struct MathNAryMaxTag {
        struct EvalBounds {
            template<typename B1, typename... Bs> struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::maximum(B1::lb, Bs::lb...);
                static constexpr std::int64_t ub = fixed_point_bounds::maximum(B1::ub, Bs::ub...);
            };
        };

        static const char* name() noexcept {
            return "max";
        }

        template<typename Context> IVARP_HD static EnableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(typename Context::NumberType a1) noexcept {
            return a1;
        }
        template<typename Context, typename A1> IVARP_H static DisableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(A1&& a1) {
            return ivarp::forward<A1>(a1);
        }

        template<typename Context> static IVARP_HD
            EnableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(typename Context::NumberType a1, typename Context::NumberType a2) noexcept
        {
            return maximum(a1, a2);
        }

        template<typename Context, typename A1, typename A2> static IVARP_H DisableForCudaNT<typename Context::NumberType, typename Context::NumberType> eval(A1&& a1, A2&& a2)
        {
            return maximum(ivarp::forward<A1>(a1), ivarp::forward<A2>(a2));
        }

        template<typename Context, typename A1, typename A2, typename A3, typename... Args>
            static IVARP_HD EnableForCudaNT<typename Context::NumberType,typename Context::NumberType> eval(A1&& a1, A2&& a2, A3&& a3, Args&&... args) noexcept
        {
            return eval<Context>(ivarp::forward<A1>(a1), eval<Context>(ivarp::forward<A2>(a2), ivarp::forward<A3>(a3), ivarp::forward<Args>(args)...));
        }

        template<typename Context, typename A1, typename A2, typename A3, typename... Args>
            static IVARP_H DisableForCudaNT<typename Context::NumberType,typename Context::NumberType> eval(A1&& a1, A2&& a2, A3&& a3, Args&&... args)
        {
            return eval<Context>(ivarp::forward<A1>(a1), eval<Context>(ivarp::forward<A2>(a2), ivarp::forward<A3>(a3), ivarp::forward<Args>(args)...));
        }
    };

    /// The MathExpression type of an n-ary operator, if at least one argument is a math expression and all
    /// arguments are math expressions, numbers or integers.
    template<typename Tag, bool Ok, typename... Args> struct NAryOpResult;
    template<typename Tag, typename... Args> struct NAryOpResult<Tag, true, Args...> {
        using Type = MathNAry<Tag, EnsureExpr<Args>...>;
    };

    /// Shorthand for NAryOpResult; also applies decay and checks arguments.
    template<typename Tag, typename... Args> using NAryOpResultType =
        typename NAryOpResult<Tag,
                              OneOf<IsMathExpr<Args>::value...>::value &&
                              AllOf<IsExprOrConstant<Args>::value...>::value,
                              BareType<Args>...>::Type;

    /// MathExpression for minimum of k >= 1 arguments.
    template<typename A1, typename... Args> static inline NAryOpResultType<MathNAryMinTag, A1, Args...>
        minimum(A1&& a1, Args&&... args)
    {
        return {ivarp::forward<A1>(a1), ivarp::forward<Args>(args)...};
    }

    /// MathExpression for maximum of k >= 1 arguments.
    template<typename A1, typename... Args> static inline NAryOpResultType<MathNAryMaxTag, A1, Args...>
        maximum(A1&& a1, Args&&... args)
    {
        return {ivarp::forward<A1>(a1), ivarp::forward<Args>(args)...};
    }
}
