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
// Created by Phillip Keldenich on 26.10.19.
//

#pragma once

#include "predicate_eval/fwd.hpp"
#include "predicate_eval/basic.hpp"
#include "predicate_eval/unary.hpp"
#include "predicate_eval/binary.hpp"
#include "predicate_eval/n_ary.hpp"
#include "eval/if_then_else.hpp"
#include "eval/custom.hpp"
#include "predicate_eval/custom.hpp"

/**
 * @file predicate_eval.hpp
 * Contains the implementation of evaluation for predicates.
 */

namespace ivarp {
namespace impl {
    template<typename Context, typename CalledType, typename... Args>
        IVARP_HD static inline EnableForCudaNT<typename Context::NumberType, PredicateEvalResultType<Context>>
            predicate_evaluate(const CalledType& c, Args&&... args)
    {
        const Array<typename Context::NumberType, sizeof...(args)> a{
            convert_number<typename Context::NumberType>(std::forward<Args>(args))...
        };
        return PredicateEvaluateImpl<Context, BareType<CalledType>, BareType<decltype(a)>>::eval(c, a);
    }

    template<typename Context, typename CalledType, typename... Args>
        IVARP_H static inline DisableForCudaNT<typename Context::NumberType, PredicateEvalResultType<Context>>
            predicate_evaluate(const CalledType& c, Args&&... args)
    {
        const Array<typename Context::NumberType, sizeof...(args)> a{
            convert_number<typename Context::NumberType>(std::forward<Args>(args))...
        };
        return PredicateEvaluateImpl<Context, BareType<CalledType>, BareType<decltype(a)>>::eval(c, a);
    }
}
}

/// Implementation of the call operator with explicit context in MathPredBase.
template<typename Derived> template<typename Context, typename... Args>
    auto ivarp::MathPredBase<Derived>::evaluate(Args&&... args) const ->
        impl::PredicateEvalResultType<Context>
{
    return impl::predicate_evaluate<Context>(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
}

template<typename Derived> template<typename Context, typename ArgArray>
    auto ivarp::MathPredBase<Derived>::array_evaluate(const ArgArray& args) const noexcept ->
        EnableForCudaNT<typename Context::NumberType, impl::PredicateEvalResultType<Context>>
{
    return impl::PredicateEvaluateImpl<Context, Derived, BareType<ArgArray>>::
        eval(static_cast<const Derived&>(*this), args);
}

template<typename Derived> template<typename Context, typename ArgArray>
    auto ivarp::MathPredBase<Derived>::array_evaluate(const ArgArray& args) const ->
        DisableForCudaNT<typename Context::NumberType, impl::PredicateEvalResultType<Context>>
{
    return impl::PredicateEvaluateImpl<Context, Derived, BareType<ArgArray>>::
        eval(static_cast<const Derived&>(*this), args);
}

/// Implementation of the call operator for evaluation with default context in MathPredBase.
template<typename Derived> template<typename... Args, typename V>
    auto ivarp::MathPredBase<Derived>::operator()(Args&&... args) const ->
        impl::PredicateEvalResultType<DefaultEvaluationContext<Derived, BareType<Args>...>>
{
    using Context = DefaultEvaluationContext<Derived, BareType<Args>...>;
    return impl::predicate_evaluate<Context>(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
}
