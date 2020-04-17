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
// Created by Phillip Keldenich on 25.10.19.
//

#pragma once

/**
 * \file has_interval_constants.hpp 
 * Implementation of the HasIntervalConstants metafunction 
 * which checks whether any interval
 * constants are contained in the given expression or predicate.
 */

namespace ivarp {
namespace impl {
    /// Default implementation: Check children.
    template<typename ExprType, typename... ChildValues> struct HasIntervalConstantsMetaTagImpl {
        using Type = OneOf<ChildValues::value...>;
    };

    /// Implementation for constants.
    template<std::int64_t LB, std::int64_t UB>
        struct HasIntervalConstantsMetaTagImpl<MathCudaConstant<LB,UB>>
    {
        using Type = std::true_type;
    };
    template<typename T, std::int64_t LB, std::int64_t UB>
        struct HasIntervalConstantsMetaTagImpl<MathConstant<T,LB,UB>>
    {
        using Type = IsIntervalType<T>;
    };
    template<typename T, bool LB, bool UB>
        struct HasIntervalConstantsMetaTagImpl<MathBoolConstant<T,LB,UB>>
    {
        using Type = std::is_same<T,IBool>;
    };

    struct HasIntervalConstantsMetaTag {
        template<typename T, typename... Children> using Eval = HasIntervalConstantsMetaTagImpl<T,Children...>;
    };

    template<typename MathExprOrPred> struct HasIntervalConstants :
        MathMetaEval<HasIntervalConstantsMetaTag, MathExprOrPred>::Type
    {
        static_assert(IsMathExprOrPred<MathExprOrPred>::value, "Invalid argument to HasIntervalConstants!");
    };
}
}
