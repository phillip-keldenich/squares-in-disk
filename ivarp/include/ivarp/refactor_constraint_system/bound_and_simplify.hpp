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
// Created by Phillip Keldenich on 17.01.20.
//

#pragma once
#include "ivarp/math_fn.hpp"
#include "ivarp/compile_time_bounds.hpp"

namespace ivarp {
namespace impl {
    template<typename MathExprOrPred, typename BoundTuple, typename Enabler = void> struct BoundAndSimplify;
}
}

#include "bound_and_simplify/bounded.hpp"
#include "bound_and_simplify/constant.hpp"
#include "bound_and_simplify/tag_based.hpp"
#include "bound_and_simplify/args.hpp"
#include "bound_and_simplify/custom.hpp"

namespace ivarp {
namespace impl {
    template<typename MathExprOrPred,
             std::enable_if_t<IsMathPred<MathExprOrPred>::value, int> = 0,
             std::enable_if_t<MathExprOrPred::lb && MathExprOrPred::ub, int> = 0>
    static inline IVARP_H auto 
        bound_and_simplify_const_pred(MathExprOrPred&&)
    {
        return MathBoolConstant<bool, true, true>{true};
    }

    template<typename MathExprOrPred,
             std::enable_if_t<IsMathPred<MathExprOrPred>::value, int> = 0,
             std::enable_if_t<!MathExprOrPred::lb && !MathExprOrPred::ub, int> = 0>
    static inline IVARP_H auto
        bound_and_simplify_const_pred(MathExprOrPred&&)
    {
        return MathBoolConstant<bool, false, false>{false};
    }

    template<typename MathExprOrPred,
             std::enable_if_t<!IsMathPred<MathExprOrPred>::value ||
                              (MathExprOrPred::lb != MathExprOrPred::ub), int> = 0>
    static inline IVARP_H std::remove_reference_t<MathExprOrPred>&&
        bound_and_simplify_const_pred(MathExprOrPred&& e)
    {
        static_assert(!std::is_lvalue_reference<MathExprOrPred>::value,
                      "lvalue passed to bound_and_simplify_const_pred!");
        return ivarp::move(e);
    }
}

    template<typename BoundTuple, typename MathExprOrPred>
    static inline IVARP_H auto bound_and_simplify(MathExprOrPred&& p) {
        static_assert(!std::is_lvalue_reference<MathExprOrPred>::value, "Non-rvalue passed into bound_and_simplify!");
        return impl::bound_and_simplify_const_pred(
            impl::BoundAndSimplify<BareType<MathExprOrPred>, BoundTuple>::apply(std::forward<MathExprOrPred>(p))
        );
    };
}
