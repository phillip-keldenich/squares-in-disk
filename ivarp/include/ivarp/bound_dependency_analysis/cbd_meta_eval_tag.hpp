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
// Created by Phillip Keldenich on 28.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    /**
     * @brief Implementation of the meta evaluation tag (see #MathMetaEval)
     * for the #compile_bound_dependencies() metafunction.
     * This is the default implementation which simply checks whether there is any dependency at all.
     *
     * @tparam ArgIndex
     * @tparam MathExprOrPred
     * @tparam ChildValues
     */
     template<std::size_t ArgIndex, typename MathExprOrPred, typename... ChildValues>
        struct ComputeBoundDepsMetaEvalTagImpl
    {
    private:
        static constexpr bool any_dependency = OneOf<
                (ChildValues::lb_depends_on_lb ||
                 ChildValues::lb_depends_on_ub ||
                 ChildValues::ub_depends_on_lb ||
                 ChildValues::ub_depends_on_ub)...>::value;

    public:
        using Type = std::conditional_t<any_dependency, CBDDefaultDependencyType, CBDNoDependencyType>;
    };

    /**
     * The meta evaluation tag (see #MathMetaEval) for the #compile_bound_dependencies() metafunction.
     * @tparam ArgIndex The index of the argument to compute dependencies for.
     */
    template<std::size_t ArgIndex> struct ComputeBoundDepsMetaEvalTag {
        template<typename MathExprOrPred, typename... ChildValues> using Eval =
            ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathExprOrPred, ChildValues...>;
    };
}
}
