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
// Created by Phillip Keldenich on 29.10.19.
//

#pragma once

namespace ivarp {
    /// Basic implementation for math predicates, providing the symbolic and evaluation call operators.
    template<typename Derived> struct MathPredBase : MathPred {
        /// Symbolic call operator.
        template<typename... Args, typename = std::enable_if_t<impl::IsSymbolicCall<Derived, Args...>::value, void>>
            inline impl::SymbolicCallResultType<Derived, typename impl::SymbolicPrepareArgs<Derived, Args...>::ArgTuple>
                operator()(Args&&... args) const;

        /// Evaluation call operator.
        template<typename... Args, typename = std::enable_if_t<impl::IsEvaluationCall<Derived, Args...>::value, void>>
            inline impl::PredicateEvalResultType<DefaultEvaluationContext<Derived, BareType<Args>...>>
                operator()(Args&&... args) const;

        /// Evaluation with user-defined Context.
        template<typename Context, typename... Args>
            inline impl::PredicateEvalResultType<Context> evaluate(Args&&... args) const;

        /// Evaluation with user-defined Context and arguments as array.
        template<typename Context, typename ArgArray>
            IVARP_HD inline EnableForCudaNT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> array_evaluate(const ArgArray& args) const noexcept;
        template<typename Context, typename ArgArray>
            IVARP_H inline DisableForCudaNT<typename Context::NumberType, impl::PredicateEvalResultType<Context>> array_evaluate(const ArgArray& args) const;
    };

    /// Marking predicate tags that are junctors (not, or, and, xor).
    struct MathPredJunctor {};

    /// Marking predicate tags that are terms (such as fn1 <= fn2).
    struct MathPredTerm {};

    /// Marking predicate junctors that are sequencing (&&, ||), i.e., where the left side is evaluated first
    /// and the right is not evaluated if the left side already determines the result.
    struct MathPredSeq {};
}
