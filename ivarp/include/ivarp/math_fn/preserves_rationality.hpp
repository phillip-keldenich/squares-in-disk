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
// Created by Phillip Keldenich on 24.10.19.
//

#pragma once

/**
 * @file preserves_rationality.hpp
 * Implementation of the PreservesRationality metafunction.
 */

namespace ivarp {
namespace impl {
    /// Metafunction that checks whether all given arguments preserve rationality.
    /// Default implementation: Check whether all children preserve rationality.
    template<typename MathExprOrPred, typename... ChildValues> struct PreservesRationalityMetaTagImpl {
        using Type = AllOf<ChildValues::value...>;
    };

    /// Implementation for unary expressions: Check argument and tag.
    template<typename Tag, typename Arg, typename ArgValue>
        struct PreservesRationalityMetaTagImpl<MathUnary<Tag,Arg>, ArgValue>
    {
        using Type = std::integral_constant<bool, ArgValue::value && !IsIrrationalTag<Tag>::value>;
    };

    /// Implementation for binary expressions: Check arguments and tag.
    template<typename Tag, typename Arg1, typename Arg2, typename A1Value, typename A2Value>
        struct PreservesRationalityMetaTagImpl<MathBinary<Tag, Arg1, Arg2>, A1Value, A2Value>
    {
        using Type = std::integral_constant<bool, A1Value::value && A2Value::value && !IsIrrationalTag<Tag>::value>;
    };

    /// Implementation for if: check args and also check the predicate for interval constants.
    template<typename Cond, typename Then, typename Else, typename CV, typename TV, typename EV>
        struct PreservesRationalityMetaTagImpl<MathTernary<MathTernaryIfThenElse, Cond, Then, Else>, CV, TV, EV>
    {
        using Type = std::integral_constant<bool, CV::value && TV::value && EV::value &&
                                                  !HasIntervalConstants<Cond>::value>;
    };

    /// Determine the resulting type of calling the eval function of a functor with a Rational number context,
    /// rational numbers as arguments for expressions and bool as arguments for predicates.
    template<typename Fn, typename... Args> struct CallWithRationalAndBool {
    private:
        using Context = DefaultContextWithNumberType<Rational>;

        static bool make_declvalue(const MathPred&);
        static Rational make_declvalue(const MathExpression&);

    public:
        using Type = BareType<decltype(
            std::declval<const Fn>().template eval<Context>(make_declvalue(std::declval<const Args>())...))>;
    };

    /// Implementation for custom functions and predicates.
    template<typename Fn, typename... Children, typename... ChildValues>
        struct PreservesRationalityMetaTagImpl<MathCustomFunction<Fn, Children...>, ChildValues...>
    {
        using Type = std::integral_constant<bool, AllOf<ChildValues::value...>::value &&
                         !IsIntervalType<typename CallWithRationalAndBool<Fn, Children...>::Type>::value>;
    };

    template<typename Fn, typename... Children, typename... ChildValues>
        struct PreservesRationalityMetaTagImpl<MathCustomPredicate<Fn, Children...>, ChildValues...>
    {
        using Type = std::integral_constant<bool, AllOf<ChildValues::value...>::value &&
            !IsIntervalType<typename CallWithRationalAndBool<Fn, Children...>::Type>::value>;
    };

    struct PreservesRationalityMetaTag {
        template<typename MathExprOrPred, typename... ChildValues>
            using Eval = PreservesRationalityMetaTagImpl<MathExprOrPred, ChildValues...>;
    };

    /// Forward to the implementation.
    template<typename MathExprOrPred> struct PreservesRationality :
        MathMetaEval<PreservesRationalityMetaTag, MathExprOrPred>::Type
    {};
}
}
