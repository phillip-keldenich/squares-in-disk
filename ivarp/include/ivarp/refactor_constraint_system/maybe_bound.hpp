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

#include "ivarp/bound_direction.hpp"

namespace ivarp {
    /**
     * @struct MaybeBound
     * @brief A bound with validity and direction depending on a check that can only be done at runtime.
     *
     * For variables x and y and some expression c, an example of this would be:
     * Constraint given: x <= c*y.
     * If c > 0: y >= x/c.
     * If c < 0: y <= x/c.
     * If c mixed: No valid bound on y.
     * Therefore, if the sign of c is not known at compile-time, this bound would not be created
     * during the compile-time only bound generation process and would be a MaybeBound on y.
     *
     * CheckDirection must behave as follows.
     * \code
     * struct CheckDirection {
     *     // Determine which bounds of the argument indicated by TargetArgIndex are used by the direction check.
     *     template<std::size_t TargetArgIndex> static constexpr BoundID compute_bound_dependencies();
     *
     *     // Return the direction of the bounds.
     *     template<typename Context, typename ArgBounds> BoundDirection check(const ArgBounds& bounds) const;
     * };
     * \endcode
     */
    template<typename BoundFunction, typename CheckDirection> struct MaybeBound {
        IVARP_DEFAULT_CM(MaybeBound);

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(typename Context, std::size_t TargetArg, typename BoundArr),
            typename Context::NumberType,
            BoundID apply_to(BoundArr& args) const {
                BoundDirection dir = direction.template check<Context>(const_cast<const BoundArr&>(args));
                if(dir == BoundDirection::NONE) {
                    return BoundID::NONE;
                }

                auto interval = bound.template array_evaluate<Context>(args);
                if(fixed_point_bounds::possibly_undefined<BoundFunction>(interval)) {
                    return BoundID::NONE;
                }

                auto& target = args[TargetArg];
                BoundID lbchanged = BoundID::NONE;
                BoundID ubchanged = BoundID::NONE;
                if((dir & BoundDirection::LEQ) != BoundDirection::NONE) {
                    if(interval.ub() < target.ub()) {
                        target.set_ub(interval.ub());
                        ubchanged = BoundID::UB;
                    }
                }
                if((dir & BoundDirection::GEQ) != BoundDirection::NONE) {
                    if(interval.lb() > target.lb()) {
                        target.set_lb(interval.lb());
                        lbchanged = BoundID::LB;
                    }
                }
                if(target.ub() < target.lb()) {
                    return BoundID::EMPTY;
                }
                return static_cast<BoundID>(static_cast<unsigned>(lbchanged) | static_cast<unsigned>(ubchanged));
            }
        )

        BoundFunction bound;
        CheckDirection direction;
        static constexpr bool success = false;
        static constexpr bool runtime_success = true;
    };

    template<typename CheckDirection> struct NegateDirectionWrapper {
        explicit NegateDirectionWrapper(const CheckDirection& c) :
            base(c)
        {}

        explicit NegateDirectionWrapper(CheckDirection&& c) noexcept :
            base(ivarp::move(c))
        {}

        template<std::size_t TargetArgIndex> static constexpr BoundID compute_bound_dependencies() {
            return CheckDirection::template compute_bound_dependencies<TargetArgIndex>();
        }

        template<typename Context, typename ArgBounds> BoundDirection check(const ArgBounds& bounds) const {
            BoundDirection res = base.template check<Context>(bounds);
            return res == BoundDirection::LEQ ? BoundDirection::GEQ :
                   res == BoundDirection::GEQ ? BoundDirection::LEQ :
                   BoundDirection::NONE;
        }

    private:
        CheckDirection base;
    };

    template<typename BoundFunction, typename CheckDirection>
        static inline auto negate_bound(const MaybeBound<BoundFunction,CheckDirection>& b)
    {
        using NegatedDirection = NegateDirectionWrapper<CheckDirection>;
        return MaybeBound<BoundFunction, NegatedDirection>{b.bound, NegatedDirection{b.direction}};
    }

    template<typename BoundFunction, typename CheckDirection>
        static inline auto negate_bound(MaybeBound<BoundFunction,CheckDirection>&& b) noexcept
    {
        using NegatedDirection = NegateDirectionWrapper<CheckDirection>;
        return MaybeBound<BoundFunction, NegatedDirection>{ivarp::move(b.bound), NegatedDirection{ivarp::move(b.direction)}};
    }
}
