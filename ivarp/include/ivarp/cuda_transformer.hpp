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
// Created by Phillip Keldenich on 13.12.19.
//

#pragma once

#include "ivarp/math_fn.hpp"
#include "ivarp/constant_folding.hpp"

namespace ivarp {
namespace impl {
    struct CUDATransformerMetaTag {};
}

    template<typename T, std::int64_t LB, std::int64_t UB>
        struct MathMetaFn<impl::CUDATransformerMetaTag, MathConstant<T,LB,UB>>
    {
        using OldType = MathConstant<T, LB, UB>;
        using Type = MathCUDAConstant<LB,UB>;

        template<typename D>
        static IVARP_H Type apply(const OldType& old, const D*) noexcept {
            return Type{old};
        }
    };

    template<typename Arg, std::int64_t LB, std::int64_t UB>
        struct MathMetaFn<impl::CUDATransformerMetaTag, ConstantFoldedExpr<Arg,LB,UB>>
    {
        using Type = MathCUDAConstant<LB, UB>;
        using OldType = ConstantFoldedExpr<Arg, LB, UB>;

        template<typename D>
        static IVARP_H Type apply(const OldType& old, const D*) noexcept {
            return Type{old};
        }
    };

    template<typename Arg, bool LB, bool UB>
        struct MathMetaFn<impl::CUDATransformerMetaTag, ConstantFoldedPred<Arg,LB,UB>>
    {
        using OldType = ConstantFoldedPred<Arg,LB,UB>;
        using Type = MathBoolConstant<typename OldType::ValueType, LB, UB>;

        template<typename D>
        static IVARP_H Type apply(const OldType& old, const D*) noexcept {
            return Type{old.value};
        }
    };

    template<typename MathExprOrPred> static inline IVARP_H auto transform_for_cuda(const MathExprOrPred& fn) {
        const auto folded = fold_constants(fn);
        return apply_metafunction<impl::CUDATransformerMetaTag>(folded);
    }
}
