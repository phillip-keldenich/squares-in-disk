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
// Created by Phillip Keldenich on 22.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    // Forward-declare all rewriters here.
    struct RelationalOpRewriter;
    struct NegationRewriter;

    template<typename Rewriter, std::size_t TargetArg, typename Constraint, typename ArgBounds>
        struct RewriterIsApplicable
    {
        static constexpr bool value = Rewriter::template IsApplicable<TargetArg, Constraint, ArgBounds>::value;
    };

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds> struct ApplicableRewriters {
        using StrippedConstraint = StripBounds<Constraint>;

        template<typename Rewriter> using RewriterApplicable =
            RewriterIsApplicable<Rewriter, TargetArg, StrippedConstraint, ArgBounds>;

        // Add all rewriters into this.
        using Result = FilterArgsType<Tuple, RewriterApplicable, RelationalOpRewriter, NegationRewriter>;
    };

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds, bool Maybe, typename ApplicableTuple>
        struct ApplyRewritersImpl;

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds, bool Maybe>
        struct ApplyRewritersImpl<TargetArg, Constraint, ArgBounds, Maybe, Tuple<>>
    {
        using BoundTuple = Tuple<>;
        static inline auto make_bound_tuple(const Constraint&) noexcept {
            return BoundTuple{};
        }
    };

    // Interpret the result from the rewriter; add the result to the tuple and/or
    // continue with the next rewriter as appropriate.
    template<std::size_t TargetArg, typename Constraint, typename ArgBounds,
             bool Maybe, bool Include, bool UseFurther, typename CurrResult, typename Curr, typename... Rest>
        struct ApplyRewritersTail;

    // The case where rewriting failed (or produced a MaybeBound we were not interested in).
    template<std::size_t TargetArg, typename Constraint, typename ArgBounds,
             bool Maybe, typename CurrResult, typename Next, typename... Rest>
        struct ApplyRewritersTail<TargetArg, Constraint, ArgBounds, Maybe, false, true, CurrResult, Next, Rest...>
    {
        using ToNext = ApplyRewritersImpl<TargetArg, Constraint, ArgBounds, Maybe, Tuple<Rest...>>;
        using BoundTuple = typename ToNext::BoundTuple;
        static inline auto make_bound_tuple(const Constraint& c) {
            return ToNext::make_bound_tuple(c);
        }
    };

    // The case where rewriting succeeded and we do not want to continue.
    template<std::size_t TargetArg, typename Constraint, typename ArgBounds,
             bool Maybe, typename CurrResult, typename Next, typename... Rest>
        struct ApplyRewritersTail<TargetArg, Constraint, ArgBounds, Maybe, true, false, CurrResult, Next, Rest...>
    {
        using BoundTuple = Tuple<typename CurrResult::BoundType>;
        static inline auto make_bound_tuple(const Constraint& c) {
            return BoundTuple(CurrResult::template make_bound<ArgBounds>(c));
        }
    };

    // The case where rewriting succeeded and we want to continue.
    template<std::size_t TargetArg, typename Constraint, typename ArgBounds,
             bool Maybe, typename CurrResult, typename Next, typename... Rest>
        struct ApplyRewritersTail<TargetArg, Constraint, ArgBounds, Maybe, true, true, CurrResult, Next, Rest...>
    {
    private:
        using ToNext = ApplyRewritersImpl<TargetArg, Constraint, ArgBounds, Maybe, Tuple<Rest...>>;

    public:
        static inline auto make_bound_tuple(const Constraint& c) {
            auto other_results = ToNext::make_bound_tuple(c);
            return prepend_tuple(ivarp::move(other_results), CurrResult::template make_bound<ArgBounds>(c));
        }
        using BoundTuple = BareType<decltype(make_bound_tuple(std::declval<Constraint>()))>;
    };

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds,
             bool Maybe, typename Next, typename... Rest>
        struct ApplyRewritersImpl<TargetArg, Constraint, ArgBounds, Maybe, Tuple<Next, Rest...>>
    {
        using NextApplicationResult = typename Next::template Apply<TargetArg, Constraint, ArgBounds>;
        static constexpr bool compile_time_success = NextApplicationResult::BoundType::success;
        static constexpr bool maybe_success = NextApplicationResult::BoundType::runtime_success;
        static constexpr bool include = compile_time_success || (Maybe && maybe_success);
        static constexpr bool use_further = !include || NextApplicationResult::try_other_rewriters;

        using Tail = ApplyRewritersTail<TargetArg, Constraint, ArgBounds,
                                        Maybe, include, use_further, NextApplicationResult, Next, Rest...>;
        using BoundTuple = typename Tail::BoundTuple;
        static inline auto make_bound_tuple(const Constraint& c) {
            return Tail::make_bound_tuple(c);
        }
    };

    template<std::size_t TargetArg, typename Constraint, typename ArgBounds, bool IncludeMaybe> struct ApplyRewriters {
    private:
        using ApplicableTuple = typename ApplicableRewriters<TargetArg, Constraint, ArgBounds>::Result;
        using Impl = ApplyRewritersImpl<TargetArg, Constraint, ArgBounds, IncludeMaybe, ApplicableTuple>;

    public:
        using BoundTuple = typename Impl::BoundTuple;
        static inline auto make_bound_tuple(const Constraint& c) {
            return Impl::make_bound_tuple(c);
        }
    };

    template<std::size_t TargetArg, typename ArgBounds, typename Constraint>
        static inline IVARP_H auto rewrite_to_bound(const Constraint& constr)
    {
        using Applier = ApplyRewriters<TargetArg, Constraint, ArgBounds, false>;
        return Applier::make_bound_tuple(constr);
    }

    template<std::size_t TargetArg, typename ArgBounds, typename Constraint>
        struct SuccessfulCTRewrite : std::integral_constant<bool, !std::is_same<
            decltype(rewrite_to_bound<TargetArg, ArgBounds>(std::declval<Constraint>())), Tuple<>
        >::value>
    {};

    template<std::size_t TargetArg, typename ArgBounds, typename Constraint>
        static inline IVARP_H auto rewrite_to_bound_with_maybe(const Constraint& constr)
    {
        using Applier = ApplyRewriters<TargetArg, Constraint, ArgBounds, true>;
        return Applier::make_bound_tuple(constr);
    }
}
}

#include "rewriters/relational_ops.hpp"
#include "rewriters/negation.hpp"
