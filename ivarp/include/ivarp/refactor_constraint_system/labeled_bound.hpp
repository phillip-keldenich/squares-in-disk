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
// Created by Phillip Keldenich on 30.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename BoundType, std::size_t TargetArg> struct LabeledBound : BoundType {
        static constexpr std::size_t target = TargetArg;

        IVARP_DEFAULT_CM(LabeledBound);
        explicit LabeledBound(BoundType&& bt) noexcept :
            BoundType(ivarp::move(bt))
        {}
        explicit LabeledBound(const BoundType& bt) :
            BoundType(bt)
        {}

        /**
         * @brief Apply this bound function to TargetArg in the given tuple or array of bounds.
         * @tparam Context
         * @tparam BoundTupOrArr
         * @param bounds
         * @return BoundID identifying which bounds of TargetArg, if any, were changed.
         */
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(typename Context, typename BoundArr),
            typename Context::NumberType,
            BoundID apply(BoundArr& bounds) const {
                return this->template apply_to<Context, TargetArg>(bounds);
            }
        )

        std::size_t get_target() const noexcept {
            return target;
        }

        const BoundType& get_bound() const noexcept {
            return *this;
        }

        BoundType& get_bound() noexcept {
            return *this;
        }
    };

    template<std::size_t ArgIndex, typename BoundType>
        static inline auto label_bound(BoundType&& bound)
    {
        return LabeledBound<BareType<BoundType>, ArgIndex>{ivarp::forward<BoundType>(bound)};
    }
}
}

