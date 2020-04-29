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
    /// Metafunction that recursively applies constant folding.
    template<typename MathExprOrPred, bool CanFold = CanConstantFold<MathExprOrPred>::value>
        struct ConstantFoldingApplyImpl;

    /// Constants themselves are not folded, even though they are theoretically foldable.
    template<typename T, std::int64_t LB, std::int64_t UB> struct ConstantFoldingApplyImpl<MathConstant<T,LB,UB>,true> {
        using Type = MathConstant<T,LB,UB>;
        static const Type& apply(const Type& m) noexcept {
            return m;
        }
    };
    template<typename T, bool LB, bool UB> struct ConstantFoldingApplyImpl<MathBoolConstant<T,LB,UB>,true> {
        using Type = MathBoolConstant<T,LB,UB>;
        static const Type&  apply(const Type& m) noexcept {
            return m;
        }
    };
    template<std::int64_t LB, std::int64_t UB> struct ConstantFoldingApplyImpl<MathCUDAConstant<LB,UB>,true> {
        using Type = MathCUDAConstant<LB,UB>;
        static const Type& apply(const Type& m) noexcept {
            return m;
        }
    };

    /// Handling non-foldable expressions by recursing on their children.
    template<typename T, bool IsEoP = IsMathExprOrPred<T>::value> struct ApplyConstantFoldingIfExprOrPred {
        using Type = T;
    };
    template<typename T> struct ApplyConstantFoldingIfExprOrPred<T,true> {
        using Type = typename ConstantFoldingApplyImpl<T>::Type;
    };

    template<template<typename...> class MathExprOrPredTemplate, typename... Args>
        struct ConstantFoldingApplyImpl<MathExprOrPredTemplate<Args...>,false>
    {
        using Type = MathExprOrPredTemplate<typename ApplyConstantFoldingIfExprOrPred<Args>::Type...>;
        using OldType = MathExprOrPredTemplate<Args...>;

        static std::conditional_t<std::is_same<Type,OldType>::value, const OldType&, Type> apply(const OldType& o) {
            return do_apply(o, std::is_same<Type,OldType>{});
        }

    private:
        template<typename=void> static const Type& do_apply(const OldType &s, std::true_type /*same*/) noexcept
        {
            return s;
        }

        template<typename=void> static Type do_apply(const OldType &s, std::false_type) {
            return do_apply(s, IndexRange<0,NumChildren<OldType>::value>{});
        }

        template<std::size_t... Inds> static Type do_apply(const OldType& s, IndexPack<Inds...>) {
            return Type{
                (ConstantFoldingApplyImpl<ChildAtType<OldType, Inds>>::apply(ChildAt<OldType, Inds>::get(s)))...
            };
        }
    };

    /// Wrap constant folding around a foldable expression or predicate.
    template<typename MathExprOrPred, bool IsPred = IsMathPred<MathExprOrPred>::value>
        struct ConstantFoldingWrap
    {
        using BoundsType = CompileTimeBounds<MathExprOrPred, Tuple<>>;
        using Type = ConstantFoldedExpr<MathExprOrPred, BoundsType::lb, BoundsType::ub>;
        static Type apply(const MathExprOrPred& m) {
            return Type{m};
        }
    };
    template<typename MathPred> struct ConstantFoldingWrap<MathPred,true> {
        using BoundsType = CompileTimeBounds<MathPred, Tuple<>>;
        using Type = ConstantFoldedPred<MathPred, BoundsType::lb, BoundsType::ub>;
        static Type apply(const MathPred& m) {
            return Type{m};
        }
    };

    /// Handling foldable expressions.
    template<typename MathExprOrPred> struct ConstantFoldingApplyImpl<MathExprOrPred,true> :
        ConstantFoldingWrap<MathExprOrPred>
    {};
}

    template<typename MathExprOrPred> static inline typename impl::ConstantFoldingApplyImpl<MathExprOrPred>::Type
        fold_constants(const MathExprOrPred& e)
    {
        return impl::ConstantFoldingApplyImpl<MathExprOrPred>::apply(e);
    }
}
