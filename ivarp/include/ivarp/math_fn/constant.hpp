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

#include "constant/fwd.hpp"
#include "constant/as.hpp"
#include "constant/constant_template.hpp"
#include "constant/cuda_constant.hpp"
#include "constant/implicitly_convertible.hpp"
#include "constant/number_to_constant.hpp"
#include "constant/ensure_expr_or_pred.hpp"

namespace ivarp {
    /// Explicit conversion to MathConstant.
    template<typename NumberType> static inline NumberToConstant<NumberType> constant(NumberType&& n) {
        return NumberToConstant<NumberType>{ivarp::forward<NumberType>(n)};
    }

    /// Macro to turn a constexpr integer into a compile-time bounded constant.
#define IVARP_CEXPR_CONSTANT(x) (impl::to_constexpr_constant<BareType<decltype(x)>, (x)>())

    namespace impl {
        template<typename IntType, IntType Value> IVARP_H static inline auto
            to_constexpr_constant()
        {
            constexpr std::int64_t vb = fixed_point_bounds::int_to_fp(Value);
            return MathConstant<IntType, vb, vb>{ Value };
        }
    }
}
