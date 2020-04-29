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
    template<typename... Args> struct CountArgs {
        static constexpr std::size_t value = sizeof...(Args);
    };

    template<template<typename...> class MathExprOrPredTemplate, typename... Args>
        struct NumChildren<MathExprOrPredTemplate<Args...>>
    {
        static constexpr std::size_t value = FilterArgsType<CountArgs, IsMathExprOrPred, Args...>::value;
    };

    template<typename C, std::int64_t LB, std::int64_t UB> struct NumChildren<MathConstant<C,LB,UB>> {
        static constexpr std::size_t value = 0;
    };
    template<std::int64_t LB, std::int64_t UB> struct NumChildren<MathCUDAConstant<LB,UB>> {
        static constexpr std::size_t value = 0;
    };
    template<typename C, bool LB, bool UB> struct NumChildren<MathBoolConstant<C,LB,UB>> {
        static constexpr std::size_t value = 0;
    };

    /// Getting the child of a unary expression or predicate.
    template<typename Tag, typename C> struct ChildAt<MathUnary<Tag,C>, 0> {
        using Type = C;
        static const Type& get(const MathUnary<Tag,C>& m) noexcept {
            return m.arg;
        }
        static Type&& get(MathUnary<Tag,C>&& m) noexcept {
            return ivarp::move(m.arg);
        }
    };
    template<typename Tag, typename C> struct ChildAt<UnaryMathPred<Tag,C>, 0> {
        using Type = C;
        static const Type& get(const UnaryMathPred<Tag,C>& m) noexcept {
            return m.arg;
        }
        static Type&& get(UnaryMathPred<Tag,C>&& m) noexcept {
            return ivarp::move(m.arg);
        }
    };

    /// Getting the children of a binary expression or predicate.
    template<typename T, typename C1, typename C2, std::size_t I> struct ChildAtBinaryImpl {
        static_assert(I < 2, "Index out of range!");

        using FirstTag = std::integral_constant<std::size_t, 0>;
        using SecondTag = std::integral_constant<std::size_t, 1>;
        using ITag = std::integral_constant<std::size_t, I>;
        using Type = std::conditional_t<I == 0, C1, C2>;

        static const Type& get(const T& m) noexcept {
            return get(m, ITag{});
        }

        static const C1& get(const T& m, FirstTag) noexcept {
            return m.arg1;
        }

        static const C2& get(const T& m, SecondTag) noexcept {
            return m.arg2;
        }

        static Type&& get(T&& m) noexcept {
            return get(ivarp::move(m), ITag{});
        }

        static C1&& get(T&& m, FirstTag) noexcept {
            return ivarp::move(m.arg1);
        }

        static C2&& get(T&& m, SecondTag) noexcept {
            return ivarp::move(m.arg2);
        }
    };
    template<typename Tag, typename C1, typename C2, std::size_t I> struct ChildAt<MathBinary<Tag,C1,C2>,I> :
        ChildAtBinaryImpl<MathBinary<Tag,C1,C2>,C1,C2,I>
    {};
    template<typename Tag, typename C1, typename C2, std::size_t I> struct ChildAt<BinaryMathPred<Tag,C1,C2>,I> :
        ChildAtBinaryImpl<BinaryMathPred<Tag,C1,C2>,C1,C2,I>
    {};

    /// Getting the children of an n-ary expression or predicate.
    template<typename T, std::size_t I> struct ChildAtNAryImpl {
        using Type = TupleElementType<I, typename T::Args>;
        static const Type& get(const T& t) noexcept {
            return ivarp::template get<I>(t.args);
        }
        static Type&& get(T&& t) noexcept {
            return ivarp::template get<I>(ivarp::move(t.args));
        }
    };
    template<typename Tag, typename... Args, std::size_t I> struct ChildAt<MathNAry<Tag,Args...>,I> :
        ChildAtNAryImpl<MathNAry<Tag,Args...>, I>
    {};
    template<typename Tag, typename... Args, std::size_t I> struct ChildAt<NAryMathPred<Tag,Args...>,I> :
        ChildAtNAryImpl<NAryMathPred<Tag,Args...>,I>
    {};

    /// Getting the children of a ternary expression.
    template<typename Tag, typename C1, typename C2, typename C3, std::size_t I>
        struct ChildAt<MathTernary<Tag,C1,C2,C3>,I>
    {
    private:
        static_assert(I < 3, "Index out of range!");
        using T = MathTernary<Tag,C1,C2,C3>;
        using FirstTag = std::integral_constant<std::size_t, 0>;
        using SecondTag = std::integral_constant<std::size_t, 1>;
        using ThirdTag = std::integral_constant<std::size_t, 2>;
        using ITag = std::integral_constant<std::size_t, I>;

        static const C1& get(const T& t, FirstTag) noexcept {
            return t.arg1;
        }

        static const C2& get(const T& t, SecondTag) noexcept {
            return t.arg2;
        }

        static const C3& get(const T& t, ThirdTag) noexcept {
            return t.arg3;
        }

        static C1&& get(T&& t, FirstTag) noexcept {
            return ivarp::move(t.arg1);
        }

        static C2&& get(T&& t, SecondTag) noexcept {
            return ivarp::move(t.arg2);
        }

        static C3&& get(T&& t, ThirdTag) noexcept {
            return ivarp::move(t.arg3);
        }

    public:
        using Type = std::conditional_t<I == 0, C1, std::conditional_t<I == 1, C2, C3>>;

        static const Type& get(const T& t) noexcept {
            return get(t, ITag{});
        }

        static Type&& get(T&& t) noexcept {
            return get(ivarp::move(t), ITag{});
        }
    };

    template<typename Functor, typename... Children, std::size_t I>
        struct ChildAt<MathCustomFunction<Functor, Children...>, I> 
    {
    private:
        using CFT = MathCustomFunction<Functor, Children...>;

    public:
        using Type = TypeAt<I, Children...>;

        static const Type& get(const CFT& t) noexcept {
            return ivarp::template get<I>(t.args);
        }

        static Type&& get(CFT&& t) noexcept {
            return ivarp::template get<I>(ivarp::move(t.args));
        }
    };

    template<typename Functor, typename... Children, std::size_t I>
        struct ChildAt<MathCustomPredicate<Functor, Children...>, I>
    {
    private:
        using CFT = MathCustomPredicate<Functor, Children...>;

    public:
        using Type = TypeAt<I, Children...>;

        static const Type& get(const CFT& t) noexcept {
            return ivarp::template get<I>(t.args);
        }

        static Type&& get(CFT&& t) noexcept {
            return ivarp::template get<I>(ivarp::move(t.args));
        }
    };
}
}
