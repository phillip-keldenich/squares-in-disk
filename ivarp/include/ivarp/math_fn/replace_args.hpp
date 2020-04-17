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
#pragma once

namespace ivarp {
namespace impl {
    /// Metafunction and factory that replaces an expression's or predicate's arguments.
    template<typename OldType, typename... NewArgs> struct ReplaceArgs;

    /// Implement for all MathExprOrPredTemplates for which we can construct new instances from
    /// just the new children.
    template<template<typename...> class MathExprOrPredTemplate, typename... Args,  typename... NewArgs>
        struct ReplaceArgs<MathExprOrPredTemplate<Args...>, NewArgs...>
    {
        using OldType = MathExprOrPredTemplate<Args...>;
        using Type = MathExprOrPredTemplate<NewArgs...>;

        template<typename OT, typename... OldChildrenArgs>
            static std::enable_if_t<std::is_same<BareType<OT>, OldType>::value, Type>
                replace(OT&&, OldChildrenArgs&&... oldargs)
        {
            return Type{std::forward<OldChildrenArgs>(oldargs)...};
        }
    };

    /// For custom functions, we also have to copy/move the functor.
    template<typename Functor, typename... Args, typename NewFunctor, typename... NewArgs>
        struct ReplaceArgs<MathCustomFunction<Functor, Args...>, NewFunctor, NewArgs...>
    {
        using OldType = MathCustomFunction<Functor, Args...>;
        using Type = MathCustomFunction<NewFunctor, NewArgs...>;

        template<typename OT, typename... OldChildrenArgs>
            static std::enable_if_t<std::is_same<BareType<OT>, OldType>::value &&
                                    !std::is_reference<OT>::value, Type>
                replace(OT&& o, OldChildrenArgs&&... oldargs)
        {
            return Type{std::move(o.functor), std::forward<OldChildrenArgs>(oldargs)...};
        }

        template<typename OT, typename... OldChildrenArgs>
            static std::enable_if_t<std::is_same<BareType<OT>, OldType>::value &&
                                    std::is_reference<OT>::value, Type>
                replace(OT&& o, OldChildrenArgs&&... oldargs)
        {
            return Type{o.functor, std::forward<OldChildrenArgs>(oldargs)...};
        }
    };

    /// For custom predicates, we also have to copy/move the functor.
    template<typename Functor, typename... Args, typename NewFunctor, typename... NewArgs>
        struct ReplaceArgs<MathCustomPredicate<Functor, Args...>, NewFunctor, NewArgs...>
    {
        using OldType = MathCustomPredicate<Functor, Args...>;
        using Type = MathCustomPredicate<NewFunctor, NewArgs...>;

        template<typename OT, typename... OldChildrenArgs>
            static std::enable_if_t<std::is_same<BareType<OT>, OldType>::value &&
                                    !std::is_reference<OT>::value, Type>
                replace(OT&& o, OldChildrenArgs&&... oldargs)
        {
            return Type{std::move(o.functor), std::forward<OldChildrenArgs>(oldargs)...};
        }

        template<typename OT, typename... OldChildrenArgs>
            static std::enable_if_t<std::is_same<BareType<OT>, OldType>::value &&
                                    std::is_reference<OT>::value, Type>
                replace(OT&& o, OldChildrenArgs&&... oldargs)
        {
            return Type{o.functor, std::forward<OldChildrenArgs>(oldargs)...};
        }
    };
}
}
