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
// Created by Phillip Keldenich on 16.12.19.
//

#pragma once

namespace ivarp {

    /**
     * @struct MathMetaEval
     * @brief Apply compile-time constant meta evaluation recursively to an expression or predicate.
     *
     * MetaEvTag should behave like this:
     * \code
     *  struct MetaEvTagExample {
     *      // may also be a template using declaration
     *      template<typename MathExprOrPred, typename... ChildValues>
     *      struct Eval {
     *           using Type = // some type depending on MathExprOrPred and ChildValues;
     *                        // typically, ChildValues... will be the ::Type types for the children of this expression.
     *      };
     *  };
     * \endcode
     */
    template<typename MetaEvTag, typename MathExprOrPred> struct MathMetaEval;

    /// Default-implementation: Evaluate children and pass their results to the tag eval.
    /// ---------- Arguments and constants ----------
    template<typename MetaEvTag, typename IndexType> struct MathMetaEval<MetaEvTag, MathArg<IndexType>> {
        using FnType = MathArg<IndexType>;
        using Type = typename MetaEvTag::template Eval<FnType>::Type;
    };
    template<typename MetaEvTag, typename T, std::int64_t LB, std::int64_t UB>
        struct MathMetaEval<MetaEvTag, MathConstant<T,LB,UB>>
    {
        using FnType = MathConstant<T, LB, UB>;
        using Type = typename MetaEvTag::template Eval<FnType>::Type;
    };
    template<typename MetaEvTag, typename T, bool LB, bool UB> struct MathMetaEval<MetaEvTag, MathBoolConstant<T,LB,UB>>
    {
        using FnType = MathBoolConstant<T,LB,UB>;
        using Type = typename MetaEvTag::template Eval<FnType>::Type;
    };
    template<typename MetaEvTag, std::int64_t LB, std::int64_t UB>
        struct MathMetaEval<MetaEvTag, MathCudaConstant<LB,UB>>
    {
        using Type = typename MetaEvTag::template Eval<MathCudaConstant<LB,UB>>::Type;
    };

    /// ---------- Bounded expressions and predicates ----------
    template<typename MetaEvTag, typename Child, typename Bounds>
        struct MathMetaEval<MetaEvTag, BoundedMathExpr<Child,Bounds>>
    {
        using ChildType = typename MathMetaEval<MetaEvTag, Child>::Type;
        using Type = typename MetaEvTag::template Eval<BoundedMathExpr<Child,Bounds>, ChildType>::Type;
    };
    template<typename MetaEvTag, typename Child, typename Bounds>
        struct MathMetaEval<MetaEvTag, BoundedPredicate<Child,Bounds>>
    {
        using ChildType = typename MathMetaEval<MetaEvTag, Child>::Type;
        using Type = typename MetaEvTag::template Eval<BoundedPredicate<Child,Bounds>, ChildType>::Type;
    };

    /// ---------- Unary expressions and predicates ----------
    template<typename MetaEvTag, typename Tag, typename A1> struct MathMetaEval<MetaEvTag, MathUnary<Tag, A1>> {
        using ChildType = typename MathMetaEval<MetaEvTag, A1>::Type;
        using Type = typename MetaEvTag::template Eval<MathUnary<Tag,A1>, ChildType>::Type;
    };
    template<typename MetaEvTag, typename Tag, typename A1> struct MathMetaEval<MetaEvTag, UnaryMathPred<Tag,A1>> {
        using ChildType = typename MathMetaEval<MetaEvTag, A1>::Type;
        using Type = typename MetaEvTag::template Eval<UnaryMathPred<Tag,A1>, ChildType>::Type;
    };

    /// ---------- Binary expressions and predicates ----------
    template<typename MetaEvTag, typename Tag, typename A1, typename A2>
        struct MathMetaEval<MetaEvTag, MathBinary<Tag, A1, A2>>
    {
        using C1Type = typename MathMetaEval<MetaEvTag, A1>::Type;
        using C2Type = typename MathMetaEval<MetaEvTag, A2>::Type;
        using Type = typename MetaEvTag::template Eval<MathBinary<Tag,A1,A2>, C1Type, C2Type>::Type;
    };
    template<typename MetaEvTag, typename Tag, typename A1, typename A2>
        struct MathMetaEval<MetaEvTag, BinaryMathPred<Tag, A1, A2>>
    {
        using C1Type = typename MathMetaEval<MetaEvTag, A1>::Type;
        using C2Type = typename MathMetaEval<MetaEvTag, A2>::Type;
        using Type = typename MetaEvTag::template Eval<BinaryMathPred<Tag,A1,A2>, C1Type, C2Type>::Type;
    };

    /// ---------- Ternary expressions ----------
    template<typename MetaEvTag, typename Tag, typename A1, typename A2, typename A3>
        struct MathMetaEval<MetaEvTag, MathTernary<Tag, A1, A2, A3>>
    {
        using C1Type = typename MathMetaEval<MetaEvTag, A1>::Type;
        using C2Type = typename MathMetaEval<MetaEvTag, A2>::Type;
        using C3Type = typename MathMetaEval<MetaEvTag, A3>::Type;
        using Type = typename MetaEvTag::template Eval<MathTernary<Tag,A1,A2,A3>, C1Type, C2Type, C3Type>::Type;
    };

    /// ---------- N-ary expressions and predicates ----------
    template<typename MetaEvTag, typename Tag, typename... Args>
        struct MathMetaEval<MetaEvTag, MathNAry<Tag, Args...>>
    {
        template<typename T> using EvalChild = typename MathMetaEval<MetaEvTag, T>::Type;
        using Type = typename MetaEvTag::template Eval<MathNAry<Tag,Args...>, EvalChild<Args>...>::Type;
    };
    template<typename MetaEvTag, typename Tag, typename... Args>
        struct MathMetaEval<MetaEvTag, NAryMathPred<Tag, Args...>>
    {
        template<typename T> using EvalChild = typename MathMetaEval<MetaEvTag, T>::Type;
        using Type = typename MetaEvTag::template Eval<NAryMathPred<Tag,Args...>, EvalChild<Args>...>::Type;
    };

    /// ---------- Custom expressions and predicates ----------
    template<typename MetaEvTag, typename Functor, typename... Args>
        struct MathMetaEval<MetaEvTag, MathCustomFunction<Functor, Args...>>
    {
        template<typename T> using EvalChild = typename MathMetaEval<MetaEvTag, T>::Type;
        using Type = typename MetaEvTag::template Eval<MathCustomFunction<Functor, Args...>, EvalChild<Args>...>::Type;
    };
    template<typename MetaEvTag, typename Functor, typename... Args>
        struct MathMetaEval<MetaEvTag, MathCustomPredicate<Functor, Args...>>
    {
        template<typename T> using EvalChild = typename MathMetaEval<MetaEvTag, T>::Type;
        using Type = typename MetaEvTag::template Eval<MathCustomPredicate<Functor, Args...>, EvalChild<Args>...>::Type;
    };

    /// ---------- Folded constants ----------
    template<typename MetaEvTag, typename Child, std::int64_t LB, std::int64_t UB>
        struct MathMetaEval<MetaEvTag, ConstantFoldedExpr<Child,LB,UB>>
    {
        using ChildType = typename MathMetaEval<MetaEvTag, Child>::Type;
        using Type = typename MetaEvTag::template Eval<ConstantFoldedExpr<Child,LB,UB>, ChildType>::Type;
    };
    template<typename MetaEvTag, typename Child, bool LB, bool UB>
        struct MathMetaEval<MetaEvTag, ConstantFoldedPred<Child,LB,UB>>
    {
        using ChildType = typename MathMetaEval<MetaEvTag, Child>::Type;
        using Type = typename MetaEvTag::template Eval<ConstantFoldedPred<Child,LB,UB>, ChildType>::Type;
    };
}
