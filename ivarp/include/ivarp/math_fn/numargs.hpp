//
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
// Created by Phillip Keldenich on 2019-10-01.
//

#pragma once

/**
 * @file numargs.hpp
 * Implementation of the NumArgs metafunction.
 */

namespace ivarp {
namespace impl {
    /**
     * @brief Implementation of the meta-evaluation tag for the NumArgs metafunction,
     * which counts the number of arguments used by an expression.
     *
     * This works by taking the maximum value over all children.
     *
     * @tparam C The current node of the expression tree.
     * @tparam Children The value in the children of C in the tree.
     */
    template<typename C, typename... Children> struct NumArgsMetaEvalTagImpl {
        using Type = std::integral_constant<std::size_t, MaxOf<Children::value...>::value>;
    };

    /**
     * @brief Implementation of the meta-evaluation tag for the NumArgs metafunction,
     * for argument nodes.
     *
     * This works by extracting the argument index + 1.
     *
     * @tparam C The current node of the expression tree.
     * @tparam Children The value in the children of C in the tree.
     */
    template<typename IndexType> struct NumArgsMetaEvalTagImpl<MathArg<IndexType>> {
        using Type = std::integral_constant<std::size_t, IndexType::value + 1>;
    };

    /**
     * @struct NumArgsMetaEvalTag
     * @brief Meta evaluation tag for the NumArgs metafunction,
     * which counts the number of arguments used by an expression.
     */
    struct NumArgsMetaEvalTag {
        /**
         * Evaluate the metafunction for a given expression node C,
         * possibly using the evaluation result for its children.
         *
         * @tparam C
         * @tparam Children
         */
        template<typename C, typename... Children> struct Eval {
            using Type = typename NumArgsMetaEvalTagImpl<C,Children...>::Type;
        };
    };

    /**
     * Implementation of the NumArgs metafunction which counts the number of arguments in an expression or predicate.
     * @tparam MathExprOrPred
     */
    template<typename MathExprOrPred> struct NumArgsImpl {
        static constexpr std::size_t value = MathMetaEval<NumArgsMetaEvalTag, MathExprOrPred>::Type::value;
    };
}

    /**
     * @brief Compute the number of arguments expected by an expression or predicate.
     * @tparam MathExprOrPred The predicate or expression to count the arguments for.
     */
    template<typename MathExprOrPred> using NumArgs = impl::NumArgsImpl<BareType<MathExprOrPred>>;
}
