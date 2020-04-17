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
    /// Check whether the given type is a Tag for a relational operator that can be used to generate bounds.
    template<typename T> struct IsRelationalOp : std::false_type {};
    template<typename Tag, typename A1, typename A2> struct IsRelationalOp<BinaryMathPred<Tag, A1, A2>> {
        static constexpr bool value = TypeIn<Tag, BinaryMathPredLEQ, BinaryMathPredLT,
                                             BinaryMathPredGEQ, BinaryMathPredGT, BinaryMathPredEQ>::value;
    };

    /// Check whether the given function is the MathArg corresponding to index TargetArg, or a bounded version of it.
    template<std::size_t TargetArg, typename T, typename Enabler = void> struct IsArgImpl : std::false_type {};
    template<std::size_t TargetArg, typename ArgIndexType>
        struct IsArgImpl<TargetArg, MathArg<ArgIndexType>, std::enable_if_t<ArgIndexType::value == TargetArg>> :
            std::true_type
    {};
    template<std::size_t TargetArg, typename T> struct IsArg : IsArgImpl<TargetArg, StripBounds<T>> {};

    /// Check whether the given function is an (arithmetic, i.e. operator-) negation of another expression,
    /// or a bounded version of it.
    template<typename T> struct IsNegatedImpl : std::false_type {};
    template<typename T> struct IsNegatedImpl<MathUnary<MathOperatorTagUnaryMinus, T>> : std::true_type {};
    template<typename T> struct IsNegated : IsNegatedImpl<StripBounds<T>> {};

    /// Check whether the given function is a negated or non-negated, bounded or unbounded version of the
    /// MathArg corresponding to the given TargetArg index.
    template<std::size_t TargetArg, typename T, bool IsNeg = IsNegated<T>::value> struct IsArgOrNegatedArg;
    template<std::size_t TargetArg, typename T> struct IsArgOrNegatedArg<TargetArg, T, false> :
        IsArg<TargetArg, T>
    {};
    template<std::size_t TargetArg, typename T> struct IsArgOrNegatedArg<TargetArg, T, true> :
        IsArg<TargetArg, typename StripBounds<T>::Arg>
    {};

    /// Check whether the given predicate is a binary math predicate, and the given arg is on one side of it,
    /// either negated/bounded or not.
    template<std::size_t TargetArg, typename Constraint> struct OneSideIsArg : std::false_type {};
    template<std::size_t TargetArg, typename Tag, typename A1, typename A2>
        struct OneSideIsArg<TargetArg, BinaryMathPred<Tag, A1, A2>>
    {
        static constexpr bool value = IsArgOrNegatedArg<TargetArg, A1>::value ^
                                      IsArgOrNegatedArg<TargetArg, A2>::value;
    };

    /// Try to apply the rewriting.
    template<std::size_t TargetArg, typename T> struct ApplyRelationalOpRewriter;
    template<std::size_t TargetArg, typename Tag, typename A1, typename A2>
        struct ApplyRelationalOpRewriter<TargetArg, BinaryMathPred<Tag,A1,A2>>
    {
    private:
        static constexpr bool is_lhs = IsArgOrNegatedArg<TargetArg, A1>::value;
        using A = std::conditional_t<is_lhs, A1, A2>;
        using NA = std::conditional_t<is_lhs, A2, A1>;
        static constexpr bool is_negated = IsNegated<A>::value;
        static constexpr bool other_depends = DependsOnArgIndex<NA, TargetArg>::value;
        using STag = std::conditional_t<is_lhs, Tag, SwapSides<Tag>>;

        struct ResultNeg {
            template<typename = void> struct Lazy {
                static constexpr bool try_other_rewriters = other_depends;
                using BoundType = CompileTimeBound<typename Negate<NA>::Type, SwapSides<STag>::lhs_bound_direction>;

                template<typename ArgBounds, typename Constraint>
                    static inline auto make_bound(const Constraint& constraint)
                {
                    auto stripped = strip_bounds(constraint);
                    return BoundType(-ChildAt<decltype(stripped), (is_lhs ? 1 : 0)>::get(ivarp::move(stripped)));
                }
            };
        };

        struct ResultPos {
            template<typename = void> struct Lazy {
                static constexpr bool try_other_rewriters = other_depends;
                using BoundType = CompileTimeBound<NA, STag::lhs_bound_direction>;

                template<typename ArgBounds, typename Constraint>
                    static inline auto make_bound(const Constraint& constraint)
                {
                    auto stripped = strip_bounds(constraint);
                    auto result = BoundType{ChildAt<decltype(stripped), (is_lhs ? 1 : 0)>::get(ivarp::move(stripped))};
                    return result;
                }
            };
        };

        using LazyResult = std::conditional_t<is_negated, ResultNeg, ResultPos>;

    public:
        using Result = typename LazyResult::template Lazy<>;
    };

    /// Rewriter that turns relational predicates such as \f$a <= b\f$ into bounds.
    struct RelationalOpRewriter {
        template<std::size_t TargetArg, typename Constraint, typename ArgBounds> struct IsApplicable {
            using SC = StripBounds<Constraint>;
            static constexpr bool value = IsRelationalOp<SC>::value && OneSideIsArg<TargetArg, SC>::value;
        };

        template<std::size_t TargetArg, typename Constraint, typename ArgBounds> using Apply =
            typename ApplyRelationalOpRewriter<TargetArg, StripBounds<Constraint>>::Result;
    };
}
}
