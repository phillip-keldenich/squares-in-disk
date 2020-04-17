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

namespace ivarp {
namespace impl {
    /// Check whether the Tag has the subtype Tag::BoundAndSimplify, signalling support for applying bounds and simplification.
    template<typename Tag, typename Enabler = void> struct TagHasBoundAndSimplify : std::false_type {};
    template<typename Tag> struct TagHasBoundAndSimplify<Tag, MakeVoid<typename Tag::BoundAndSimplify>> : std::true_type {};


    /// Check whether the Tag has the subtype Tag::EvalBounds, signalling support for compile-time evaluation of bounds.
    template<typename Tag, typename Enabler = void> struct TagHasEvalBounds : std::false_type {};
    template<typename Tag> struct TagHasEvalBounds<Tag, MakeVoid<typename Tag::EvalBounds>> : std::true_type {};

    /// Check whether the Tag has the subtype Tag::BoundedEval, signalling support for runtime evaluation using compile-time bounds.
    template<typename Tag, typename Enabler = void> struct TagHasBoundedEval : std::false_type {};
    template<typename Tag> struct TagHasBoundedEval<Tag, MakeVoid<typename Tag::BoundedEval>> : std::true_type {};

    /// Check whether the tag supports bounded evaluation and all arguments have bounds
    /// (does not check for actual unboundedness, i.e., does not consider the bound values).
    template<typename Tag, typename... Args> using IsBoundedTagInvokation =
        std::integral_constant<bool, TagHasBoundedEval<Tag>::value && AllOf<IsBounded<Args>::value...>::value>;

    /// Invoke a tag for evaluation without bounds.
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(typename Tag, typename Context, typename... Args, typename ArgArray,
                              std::enable_if_t<!IsBoundedTagInvokation<Tag,Args...>::value,int> = 0),
        typename Context::NumberType,
        static inline auto invoke_tag(const ArgArray& arg_values, const Args&... args)
            noexcept(AllowsCuda<typename Context::NumberType>::value)
        {
            return Tag::template eval<Context>((PredicateEvaluateImpl<Context,Args,ArgArray>::eval(args, arg_values))...);
        }
    )

    /// Invoke a tag for evaluation with bounds.
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(typename Tag, typename Context, typename... Args, typename ArgArray,
                              std::enable_if_t<IsBoundedTagInvokation<Tag,Args...>::value,int> = 0),
        typename Context::NumberType,
        static inline auto invoke_tag(const ArgArray& arg_values, const Args&... args)
            noexcept(AllowsCuda<typename Context::NumberType>::value)
        {
            return Tag::BoundedEval::template eval<Context, Args...>(
                (PredicateEvaluateImpl<Context,Args,ArgArray>::eval(args, arg_values))...
            );
        }
    )
}
}
