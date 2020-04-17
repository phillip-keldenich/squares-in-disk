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
// Created by Phillip Keldenich on 12.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Default implementation: Check whether any child depends on the argument.
    template<typename ArgIndexType, typename MathExprOrPred, typename... ChildValues>
        struct DependsOnArgMetaEvalTagImpl
    {
        using Type = OneOf<ChildValues::value...>;
    };

    /// Implementation for arguments.
    template<typename ArgIndexType, typename AIT> struct DependsOnArgMetaEvalTagImpl<ArgIndexType, MathArg<AIT>> {
        using Type = std::integral_constant<bool, ArgIndexType::value == AIT::value>;
    };

    template<typename ArgIndexType> struct DependsOnArgMetaEvalTag {
        template<typename MathExprOrPred, typename... ChildValues> using Eval =
            DependsOnArgMetaEvalTagImpl<ArgIndexType, MathExprOrPred, ChildValues...>;
    };

    /// A metafunction that checks whether a function or predicate (symbolically) depends on a given argument.
    template<typename MathExprOrPred, typename Arg> struct DependsOnArgImpl;

    template<typename Arg, typename... MathExprsOrPreds> struct AnyDependOnArgImpl {
        static constexpr bool value = OneOf<DependsOnArgImpl<MathExprsOrPreds,Arg>::value...>::value;
    };

    /// Default implementation: Check whether any children depend on arg.
    template<template<typename...> class MathExprOrPredTemplate, typename... Args, typename Arg>
        struct DependsOnArgImpl<MathExprOrPredTemplate<Args...>, Arg>
    {
        static_assert(IsMathExprOrPred<MathExprOrPredTemplate<Args...>>::value,
                      "DependsOnArg used on non-function/non-predicate type!");
        static_assert(IsMathArg<Arg>::value, "Second argument to DependsOnArg is not a MathArg!");

        /// Bind Arg to the first argument of AnyDependOnArgImpl.
        template<typename... A> using AnyDependOnArg = AnyDependOnArgImpl<Arg, A...>;

        /// Check whether any children depend on the given argument.
        static constexpr bool value = FilterArgsType<AnyDependOnArg, IsMathExprOrPred, Args...>::value;
    };

    /// Implementation for arguments.
    template<typename Index, typename Arg> struct DependsOnArgImpl<MathArg<Index>, Arg> {
        static constexpr bool value = (Index::value == Arg::index);
    };

    template<typename MathExprOrPred, typename Arg> using DependsOnArg =
        typename MathMetaEval<DependsOnArgMetaEvalTag<BareType<Arg>>, BareType<MathExprOrPred>>::Type;

    template<typename MathExprOrPred, std::size_t ArgIndex> using DependsOnArgIndex =
        typename MathMetaEval<DependsOnArgMetaEvalTag<std::integral_constant<unsigned,ArgIndex>>,
                              BareType<MathExprOrPred>>::Type;
}
}
