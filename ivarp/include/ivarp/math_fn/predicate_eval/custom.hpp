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
// Created by Phillip Keldenich on 22.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Implementation of evaluation for custom predicates.
    template<typename Context, typename Functor, typename... Args_, typename ArgArray>
        struct PredicateEvaluateImpl<Context, MathCustomPredicate<Functor, Args_...>, ArgArray, false>
    {
        using CalledType = MathCustomPredicate<Functor,Args_...>;
        using ArgsType = typename CalledType::Args;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline auto eval(const CalledType& c, const ArgArray& args) noexcept(AllowsCuda<NumberType>::value) {
                return do_eval(c, args, IndexRange<0,sizeof...(Args_)>{});
            }
        )

    private:
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(std::size_t... Inds), NumberType,
            static inline auto do_eval(const CalledType& c, const ArgArray& args)
                noexcept(AllowsCuda<NumberType>::value)
            {
                return c.functor.template eval<Context>(
                    (PredicateEvaluateImpl<Context, TupleElementType<Inds, ArgsType>, ArgArray>::eval(
                        get<Inds>(c.args), args
                    ))...
                );
            }
        )
    };
}
}
