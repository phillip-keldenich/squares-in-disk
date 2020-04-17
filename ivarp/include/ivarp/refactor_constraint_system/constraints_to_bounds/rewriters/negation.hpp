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
// Created by Phillip Keldenich on 23.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename T> struct IsLogicNegationImpl : std::false_type {};
    template<typename T> struct IsLogicNegationImpl<UnaryMathPred<UnaryMathPredNotTag, T>> : std::true_type {};
    template<typename T> using IsLogicNegation = IsLogicNegationImpl<StripBounds<T>>;

    /// Default implementation: Empty applicable tuple.
    template<std::size_t TargetArg, typename Constraint, typename ArgBounds, typename BoundSoFar, typename ChildApplicable>
        struct ApplyNegationRewriter
    {
        using Result = BoundSoFar;
    };

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds, typename BoundSoFar, typename Curr, typename... Rest>
        struct ApplyNegationRewriter<TargetArg, Constraint, ArgBounds, BoundSoFar, Tuple<Curr,Rest...>>
    {
        using CurrApplicationResult = typename Curr::template Apply<TargetArg, Constraint, ArgBounds>;
        using CurrBounds = typename CurrApplicationResult::BoundType;
        using NegCurrBounds = BareType<decltype(negate_bound(std::declval<CurrBounds>()))>;

        static constexpr bool curr_success = NegCurrBounds::success;
        static constexpr bool curr_runtime_success = !BoundSoFar::BoundType::runtime_success && NegCurrBounds::runtime_success;

        struct CurrSuccess {
            template<typename = void> struct Lazy {
                using BoundType = NegCurrBounds;
                static constexpr bool try_other_rewriters = true;

                template<typename ArgB, typename C>
                    static inline auto make_bound(const C& constraint)
                {
                    auto stripped = strip_bounds(strip_bounds(constraint).arg1);
                    auto rewritten = CurrApplicationResult::template make_bound<ArgB>(stripped);
                    return negate_bound(ivarp::move(rewritten));
                }
            };
        };

        struct CurrRuntimeSuccess {
            template<typename = void> using Lazy = typename
                ApplyNegationRewriter<TargetArg, Constraint, ArgBounds, CurrApplicationResult, Tuple<Rest...>>::Result;
        };

        struct CurrFailure {
            template<typename = void> using Lazy = typename
                ApplyNegationRewriter<TargetArg, Constraint, ArgBounds, BoundSoFar, Tuple<Rest...>>::Result;
        };

        using CurrResult = std::conditional_t<curr_success, CurrSuccess, std::conditional_t<curr_runtime_success, CurrRuntimeSuccess, CurrFailure>>;
        using Result = typename CurrResult::template Lazy<>;
    };


    struct NegationRewriter {
        template<std::size_t TargetArg, typename Constraint, typename ArgBounds> struct IsApplicable {
            static constexpr bool value = IsLogicNegation<Constraint>::value &&
                                          DependsOnArgIndex<Constraint,TargetArg>::value;
        };

        template<typename Constraint>
            using StripNegationAndBounds = StripBounds<typename StripBounds<Constraint>::Arg1>;

        struct RewriteFailedResult {
            using BoundType = RewriteFailed;
            static constexpr bool try_other_rewriters = true;
        };

        template<std::size_t TargetArg, typename Constraint, typename ArgBounds> using Apply =
            typename ApplyNegationRewriter<
                    TargetArg, Constraint, ArgBounds, RewriteFailedResult,
                    typename ApplicableRewriters<TargetArg, StripNegationAndBounds<Constraint>, ArgBounds>::Result
            >::Result;
    };
}
}
