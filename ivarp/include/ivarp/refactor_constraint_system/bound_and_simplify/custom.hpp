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
// Created by Phillip Keldenich on 21.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename Custom, typename BoundTuple> struct BoundAndSimplifyCustom;
    template<template<typename...> class Template, typename Functor, typename... Args, typename BoundTuple>
        struct BoundAndSimplifyCustom<Template<Functor,Args...>, BoundTuple>
    {
        using OldType = Template<Functor, Args...>;
        static inline IVARP_H auto apply(OldType&& old) {
            return apply_to_children(std::move(old), IndexRange<0,sizeof...(Args)>{});
        }

    private:
        template<typename O, std::size_t I> using CAt = ChildAt<OldType, I>;
        template<typename O, std::size_t I> using BSAt =
            BoundAndSimplify<typename CAt<O,I>::Type, BoundTuple>;

        template<std::size_t... Inds>
            static inline IVARP_H auto apply_to_children(OldType&& old, IndexPack<Inds...>)
        {
            return pack_result(std::move(old.functor),
                               (BSAt<OldType,Inds>::apply(CAt<OldType,Inds>::get(std::forward<OldType>(old))))...);
        }

        struct Expr {
            template<typename Inner> using NewType = BoundedMathExpr<Inner, fixed_point_bounds::Unbounded>;
        };
        struct Pred {
            template<typename Inner> using NewType = BoundedPredicate<Inner, fixed_point_bounds::UnboundedPredicate>;
        };

        using NewTypeWrapper = std::conditional_t<IsMathExpr<OldType>::value, Expr, Pred>;
        template<typename Inner> using NewType = typename NewTypeWrapper::template NewType<Inner>;

        template<typename... NewBoundedChildren>
            static inline IVARP_H auto pack_result(Functor&& f, NewBoundedChildren&&... bounded)
        {
            static_assert(AllOf<(!std::is_lvalue_reference<NewBoundedChildren>::value)...>::value,
                          "Non-rvalue in pack_result!");
            static_assert(AllOf<IsBounded<NewBoundedChildren>::value...>::value,
                          "Unbounded child in pack_result!");

            using NewInnerType = Template<Functor, NewBoundedChildren...>;
            return NewType<NewInnerType>{NewInnerType{std::move(f), std::forward<NewBoundedChildren>(bounded)...}};
        }
    };

    template<typename Functor, typename... Args, typename BoundTuple>
        struct BoundAndSimplify<MathCustomFunction<Functor,Args...>, BoundTuple, void> :
            BoundAndSimplifyCustom<MathCustomFunction<Functor,Args...>, BoundTuple>
    {};

    template<typename Functor, typename... Args, typename BoundTuple>
        struct BoundAndSimplify<MathCustomPredicate<Functor,Args...>, BoundTuple, void> :
            BoundAndSimplifyCustom<MathCustomPredicate<Functor,Args...>, BoundTuple>
    {};
}
}

