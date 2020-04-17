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

#include <type_traits>

namespace ivarp {
namespace impl {
    /// Replace the children of a tag-based expression or predicate.
    template<typename MathExprOrPred, typename... NewBoundedChildren> struct TaggedReplaceChildren;
    template<typename Tag, template<typename...> class ExprOrPredTemplate,
        typename... OldChildren, typename... NewBoundedChildren>
        struct TaggedReplaceChildren<ExprOrPredTemplate<Tag, OldChildren...>, NewBoundedChildren...> {
        using Type = ExprOrPredTemplate<Tag, NewBoundedChildren...>;
    };

    /// Bound and simplify a tag-based expression or predicate; used in the case where the tag provides neither
    /// bound evaluation nor implements BoundAndSimplify.
    template<typename Tag, typename Enabler = void> struct BoundAndSimplifyTagged {
        struct DefaultBoundPredicate {
            template<typename T> using Wrap = BoundedPredicate<T, fixed_point_bounds::UnboundedPredicate>;

            template<typename T> static inline auto wrap(T&& t) noexcept {
                static_assert(!std::is_lvalue_reference<T>::value, "Non-rvalue passed into bound_and_simplify!");
                return Wrap<T>{std::forward<T>(t)};
            }
        };
        struct DefaultBoundExpression {
            template<typename T> using Wrap = BoundedMathExpr<T, fixed_point_bounds::Unbounded>;

            template<typename T> static inline auto wrap(T&& t) noexcept {
                static_assert(!std::is_lvalue_reference<T>::value, "Non-rvalue passed into bound_and_simplify!");
                return Wrap<T>{std::forward<T>(t)};
            }
        };

        template<typename OldType, typename... NewBoundedChildren> static inline auto bound_and_simplify(OldType&& old, NewBoundedChildren&&... children) {
            static_assert(!std::is_lvalue_reference<OldType>::value, "OldType must be an rvalue!");
            static_assert(AllOf<(!std::is_lvalue_reference<NewBoundedChildren>::value)...>::value, "NewBoundedChildren must be rvalues!");

            using RepType = typename TaggedReplaceChildren<OldType, NewBoundedChildren...>::Type;
            RepType replaced{ std::forward<NewBoundedChildren>(children)... };
            using ReplacedWrapper = std::conditional_t<IsMathExpr<RepType>::value, DefaultBoundExpression, DefaultBoundPredicate>;
            return ReplacedWrapper::wrap(ivarp::move(replaced));
        }
    };

    /// Bound and simplify a tag-based expression or predicate; used in the case where the tag provides BoundAndSimplify.
    template<typename Tag> struct BoundAndSimplifyTagged<Tag, std::enable_if_t<TagHasBoundAndSimplify<Tag>::value>> {
        template<typename OldType, typename... NewBoundedChildren> static inline auto bound_and_simplify(OldType&& old, NewBoundedChildren&&... children) {
            static_assert(!std::is_lvalue_reference<OldType>::value, "OldType must be an rvalue!");
            static_assert(AllOf<(!std::is_lvalue_reference<NewBoundedChildren>::value)...>::value, "NewBoundedChildren must be rvalues!");
            return Tag::BoundAndSimplify::bound_and_simplify(std::forward<OldType>(old), std::forward<NewBoundedChildren>(children)...);
        }
    };

    /// Bound and simplify a tag-based expression or predicate;
    /// used in the case where the tag does not have BoundAndSimplify but at least EvalBounds.
    template<typename Tag> struct BoundAndSimplifyTagged<Tag,
             std::enable_if_t<!TagHasBoundAndSimplify<Tag>::value && TagHasEvalBounds<Tag>::value>>
    {
        struct Expr {
            template<typename T, std::int64_t LB, std::int64_t UB> using Wrap = BoundedMathExpr<T, ExpressionBounds<LB,UB>>;
            template<typename BoundsType, typename Child> static inline auto wrap(Child&& c) noexcept {
                static_assert(!std::is_lvalue_reference<Child>::value, "Non-rvalue passed to wrap!");
                return Wrap<Child, BoundsType::lb, BoundsType::ub>{std::forward<Child>(c)};
            }
        };

        struct Pred {
            template<typename T, bool LB, bool UB> using Wrap = BoundedPredicate<T, PredicateBounds<LB,UB>>;
            template<typename BoundsType, typename Child> static inline auto wrap(Child&& c) noexcept {
                static_assert(!std::is_lvalue_reference<Child>::value, "Non-rvalue passed to wrap!");
                return Wrap<Child, BoundsType::lb, BoundsType::ub>{std::forward<Child>(c)};
            }
        };

        template<typename OldType, typename... NewBoundedChildren>
            static inline auto bound_and_simplify(OldType&& /*old*/, NewBoundedChildren&&... children)
        {
            static_assert(!std::is_lvalue_reference<OldType>::value, "OldType must be an rvalue!");
            static_assert(AllOf<(!std::is_lvalue_reference<NewBoundedChildren>::value)...>::value,
                          "NewBoundedChildren must be rvalues!");
            static_assert(AllOf<IsBounded<NewBoundedChildren>::value...>::value, "Unbounded children!");

            using RepType = typename TaggedReplaceChildren<OldType, NewBoundedChildren...>::Type;
            RepType replaced{std::forward<NewBoundedChildren>(children)...};
            using ReplacedWrapper = std::conditional_t<IsMathExpr<RepType>::value, Expr, Pred>;
            using EvalBounds = typename Tag::EvalBounds::template Eval<NewBoundedChildren...>;
            return ReplacedWrapper::template wrap<EvalBounds>(ivarp::move(replaced));
        }
    };

    template<typename MathExprOrPred, typename ArgBounds>
        struct BoundAndSimplify<MathExprOrPred, ArgBounds, std::enable_if_t<HasTag<MathExprOrPred>::value>>
    {
        using OldType = MathExprOrPred;
        using Tag = TagOf<MathExprOrPred>;

        static inline IVARP_H auto apply(OldType&& old) {
            using IndRange = IndexRange<0, NumChildren<OldType>::value>;
            return apply_to_children(ivarp::move(old), IndRange{});
        }

    private:
        template<std::size_t... Inds>
        static inline IVARP_H auto apply_to_children(OldType&& old, IndexPack<Inds...>) {
            return BoundAndSimplifyTagged<Tag>::bound_and_simplify(
                ivarp::move(old), (BoundAndSimplify<ChildAtType<OldType, Inds>, ArgBounds>::
                        apply(ChildAt<OldType, Inds>::get(ivarp::move(old))))...
            );
        }
    };
}
}
