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
// Created by Phillip Keldenich on 10.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Default implementation for function types that cannot handle bounds.
    template<typename VarBoundTuple, typename MathExprOrPred, typename Void, typename... ChildValues>
        struct ComputeBoundsMetaEvalTagImpl
    {
    private:
        struct TypeExpr {
            static constexpr auto lb = fixed_point_bounds::min_bound();
            static constexpr auto ub = fixed_point_bounds::max_bound();
        };
        struct TypePred {
            static constexpr bool lb = false;
            static constexpr bool ub = true;
        };

    public:
        using Type = std::conditional_t<IsMathExpr<MathExprOrPred>::value, TypeExpr, TypePred>;
    };

    /// Implementation for bounded types (constants, folded constants).
    template<typename VarBoundTuple, typename MathExprOrPred, typename... ChildValues>
        struct ComputeBoundsMetaEvalTagImpl<VarBoundTuple, MathExprOrPred,
                                            MakeVoid<std::enable_if_t<IsBounded<MathExprOrPred>::value>>,
                                            ChildValues...>
    {
        struct Type {
            static constexpr auto lb = MathExprOrPred::lb;
            static constexpr auto ub = MathExprOrPred::ub;
        };
    };

    /// Implementation for tagged types (unary, binary, ternary, n-ary expressions/predicates)
    /// with tags that can handle bounds.
    template<typename VarBoundTuple, typename MathExprOrPred, typename... ChildValues>
        struct ComputeBoundsMetaEvalTagImpl<VarBoundTuple, MathExprOrPred,
            std::enable_if_t<TagHasEvalBounds<TagOf<MathExprOrPred>>::value>,
            ChildValues...>
    {
        using Type = typename TagOf<MathExprOrPred>::EvalBounds::template Eval<ChildValues...>;
    };

    /// Implementation for arguments.
    template<typename VarBoundTuple, typename ArgIndexType>
        struct ComputeBoundsMetaEvalTagImpl<VarBoundTuple, MathArg<ArgIndexType>, void>
    {
        using Type = TupleElementType<ArgIndexType::value, VarBoundTuple>;
    };

    template<typename VarBoundTuple> struct ComputeBoundsMetaEvalTag {
        template<typename MathExprOrPred, typename... ChildValues> using Eval =
            ComputeBoundsMetaEvalTagImpl<VarBoundTuple, MathExprOrPred, void, ChildValues...>;
    };
}

    /// Compute compile-time bounds for the given math expression or predicate.
    /// The second parameter should be Tuple<VarBounds...>, where the ith entry has a type containing
    /// static constexpr std::int64_t fields lb and ub providing bounds for the value of the ith variable.
    /// For constant folding, Tuple<> is passed.
    template<typename MathExprOrPred, typename VarBoundTuple> using CompileTimeBounds =
        typename MathMetaEval<impl::ComputeBoundsMetaEvalTag<VarBoundTuple>, MathExprOrPred>::Type;
}
