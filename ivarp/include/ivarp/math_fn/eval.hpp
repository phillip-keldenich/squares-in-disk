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
// Created by Phillip Keldenich on 08.10.19.
//

#pragma once

#include "ivarp/array.hpp"
#include "eval/number_type.hpp"
#include "eval/basic.hpp"
#include "eval/unary.hpp"
#include "eval/binary.hpp"
#include "eval/ternary.hpp"
#include "eval/n_ary.hpp"

namespace ivarp {
namespace impl {
    template<typename Context, typename CalledType, typename... Args>
        EvaluationCallResultType<Context, BareType<CalledType>, Args...>
            evaluate(const CalledType& c, Args&&... args)
    {
        const Array<typename Context::NumberType, sizeof...(args)> a{
            convert_number<typename Context::NumberType>(std::forward<Args>(args))...
        };
        return EvaluateImpl<Context, BareType<CalledType>, BareType<decltype(a)>>::eval(c, a);
    }
}
}

template<typename Derived> template<typename Context, typename... Args>
    auto ivarp::MathExpressionBase<Derived>::evaluate(Args&&... args) const ->
        impl::EvaluationCallResultType<Context, Derived, Args...>
{
    return impl::evaluate<Context>(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
}

template<typename Derived> template<typename Context, typename ArrayType>
    auto ivarp::MathExpressionBase<Derived>::array_evaluate(const ArrayType &args) const noexcept ->
        EnableForCudaNT<typename Context::NumberType, impl::ArrayEvaluationCallResultType<Context, ArrayType>>
{
    return impl::EvaluateImpl<Context, Derived, BareType<ArrayType>>::
        eval(static_cast<const Derived&>(*this), args);
}

template<typename Derived> template<typename Context, typename ArrayType>
    auto ivarp::MathExpressionBase<Derived>::array_evaluate(const ArrayType &args) const ->
        DisableForCudaNT<typename Context::NumberType, impl::ArrayEvaluationCallResultType<Context, ArrayType>>
{
    return impl::EvaluateImpl<Context, Derived, BareType<ArrayType>>::
        eval(static_cast<const Derived&>(*this), args);
}

template<typename Derived> template<typename... Args, typename V>
    auto ivarp::MathExpressionBase<Derived>::operator()(Args&&... args) const ->
        impl::EvaluationCallResultType<DefaultEvaluationContext<Derived, BareType<Args>...>, Derived, Args...>
{
    using Context = DefaultEvaluationContext<Derived, BareType<Args>...>;
    return impl::evaluate<Context>(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
}
