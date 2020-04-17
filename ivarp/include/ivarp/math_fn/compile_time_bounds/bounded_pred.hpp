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
// Created by Phillip Keldenich on 10.01.20.
//

#pragma once

namespace ivarp {
    template<typename MathPred_, typename BoundsType> struct BoundedPredicate :
        MathPredBase<BoundedPredicate<MathPred_,BoundsType>>
    {
        using Child = MathPred_;

        static constexpr bool lb = BoundsType::lb;
        static constexpr bool ub = BoundsType::ub;
        static constexpr bool cuda_supported = Child::cuda_supported;

        template<typename C, std::enable_if_t<!std::is_same<BareType<C>, BoundedPredicate>::value, int> = 0>
            explicit BoundedPredicate(C&& c) :
                child(ivarp::forward<C>(c))
        {}

        Child child;
    };

    template<typename BoundType, typename MathPred_, std::enable_if_t<IsMathPred<MathPred_>::value, int> = 0>
        static inline auto known_bounds(MathPred_&& pred)
    {
        return BoundedPredicate<BareType<MathPred_>, BoundType>{ivarp::forward<MathPred_>(pred)};
    }

    template<typename T, typename BT> struct StripBoundsImpl<BoundedPredicate<T,BT>> {
    private:
        using OldType = BoundedPredicate<T,BT>;

    public:
        using Type = T;

        static inline IVARP_H Type strip_bounds(const OldType& o) {
            return o.child;
        }

        static inline IVARP_H Type strip_bounds(OldType&& o) noexcept {
            return ivarp::move(o.child);
        }
    };
}
