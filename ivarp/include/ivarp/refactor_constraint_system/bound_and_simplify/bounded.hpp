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
// Created by Phillip Keldenich on 18.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename Child, typename BoundsType, typename ArgBounds>
        struct BoundAndSimplify<BoundedMathExpr<Child, BoundsType>, ArgBounds, void>
    {
        using OldType = BoundedMathExpr<Child, BoundsType>;
        static inline IVARP_H auto apply(OldType&& old) {
            auto child_result = BoundAndSimplify<Child, ArgBounds>::apply(ivarp::move(old.child));
            using CRT = decltype(child_result);
            using NewBounds = ExpressionBounds<ivarp::max(CRT::lb, BoundsType::lb), ivarp::min(CRT::ub, BoundsType::ub)>;
            return BoundedMathExpr<typename CRT::Child, NewBounds>{ivarp::move(child_result.child)};
        }
    };

    template<typename Child, typename BoundsType, typename ArgBounds>
        struct BoundAndSimplify<BoundedPredicate<Child, BoundsType>, ArgBounds, void>
    {
    private:
        using OldType = BoundedPredicate<Child, BoundsType>;

        template<typename NewBounds, typename Ch,
                 std::enable_if_t<NewBounds::lb && NewBounds::ub, int> = 0>
            static inline IVARP_H auto simplify_by_bound(Ch&&) noexcept
        {
            return MathBoolConstant<bool,true,true>{true};
        }

        template<typename NewBounds, typename Ch,
                 std::enable_if_t<!NewBounds::lb && !NewBounds::ub, int> = 0>
            static inline IVARP_H auto simplify_by_bound(Ch&&) noexcept
        {
            return MathBoolConstant<bool,false,false>{false};
        }

        template<typename NewBounds, typename Ch,
            std::enable_if_t<!NewBounds::lb && NewBounds::ub, int> = 0>
            static inline IVARP_H auto simplify_by_bound(Ch&& c) noexcept
        {
            return BoundedPredicate<std::remove_reference_t<Ch>, fixed_point_bounds::UnboundedPredicate>{
                ivarp::forward<Ch>(c)
            };
        }

    public:
        static inline IVARP_H auto apply(OldType&& old) {
            auto child_result = BoundAndSimplify<Child, ArgBounds>::apply(ivarp::move(old.child));
            using CRT = decltype(child_result);
            using NewBounds = ExpressionBounds<CRT::lb || BoundsType::lb, CRT::ub && BoundsType::ub>;
            return simplify_by_bound<NewBounds>(ivarp::move(child_result.child));
        }
    };
}
}
