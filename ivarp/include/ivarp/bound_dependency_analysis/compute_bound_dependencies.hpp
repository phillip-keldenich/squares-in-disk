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
    template<typename BoundedMathExprOrPred, std::size_t ArgIndex>
        static inline constexpr BoundDependencies compute_bound_dependencies() noexcept
    {
        static_assert(impl::IsBounded<BoundedMathExprOrPred>::value, "Bound dependency analysis depends on bounds!");
        using ResultType = typename MathMetaEval<impl::ComputeBoundDepsMetaEvalTag<ArgIndex>,
                                                 BoundedMathExprOrPred>::Type;
        return {ResultType::lb_depends_on_lb, ResultType::lb_depends_on_ub,
                ResultType::ub_depends_on_lb, ResultType::ub_depends_on_ub};
    }
}

