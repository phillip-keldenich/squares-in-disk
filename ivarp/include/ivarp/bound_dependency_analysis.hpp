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
// Created by Phillip Keldenich on 28.01.20.
//

#pragma once

#include "ivarp/math_fn.hpp"

namespace ivarp {
    /**
     * An enum whose elements identify zero, one or two bounds of an interval.
     */
    enum class BoundID : unsigned {
        NONE = 0,
        LB = 1,
        UB = 2,
        BOTH = 3,
        EMPTY = 4 // returned from bound applications if an interval became empty
    };

    template<std::size_t TargetArg> struct LabeledBoundID {
        static constexpr std::size_t target = TargetArg;
        BoundID bound_id;
    };

    static inline constexpr bool uses_lb(BoundID b) noexcept {
        return b == BoundID::LB || b == BoundID::BOTH;
    }

    static inline constexpr bool uses_ub(BoundID b) noexcept {
        return b == BoundID::UB || b == BoundID::BOTH;
    }

    /**
     * @struct BoundDependencies
     * @brief Contains information about which bound of a given expression may depend on
     * which bound of an argument.
     *
     * Generally speaking, this is an over-approximation
     * in the sense that the flags in this struct may be set even if
     * the evaluation sometimes does not actually depend on the argument bound.
     */
    struct BoundDependencies {
        bool lb_depends_on_lb;
        bool lb_depends_on_ub;
        bool ub_depends_on_lb;
        bool ub_depends_on_ub;

        template<typename BDepType> static constexpr BoundDependencies from_type() {
            return BoundDependencies{BDepType::lb_depends_on_lb, BDepType::lb_depends_on_ub,
                                     BDepType::ub_depends_on_lb, BDepType::ub_depends_on_ub};
        }

        constexpr bool operator==(const BoundDependencies& o) const noexcept {
            return lb_depends_on_lb == o.lb_depends_on_lb &&
                   lb_depends_on_ub == o.lb_depends_on_ub &&
                   ub_depends_on_lb == o.ub_depends_on_lb &&
                   ub_depends_on_ub == o.ub_depends_on_ub;
        }

        /**
         * @brief Compute how a computation using the given bounds of its child expression
         * depends on the target argument.
         * @param used_bounds The bounds of the child expression that are used.
         * @param bd How the child bounds depend on the target argument.
         * @return
         */
        constexpr static bool depends_on_lb(BoundID used_bounds, const BoundDependencies& bd) noexcept {
            return (uses_lb(used_bounds) && bd.lb_depends_on_lb) || (uses_ub(used_bounds) && bd.ub_depends_on_lb);
        }

        /**
         * @brief Compute how a computation using the given bounds of its child expression
         * depends on the target argument.
         * @param used_bounds The bounds of the child expression that are used.
         * @param bd How the child bounds depend on the target argument.
         * @return
         */
        constexpr static bool depends_on_ub(BoundID used_bounds, const BoundDependencies& bd) noexcept {
            return (uses_lb(used_bounds) && bd.lb_depends_on_ub) || (uses_ub(used_bounds) && bd.ub_depends_on_ub);
        }

        /**
         * Compute BoundDependencies for a binary operator such as * or /.
         *
         * @param lb_bounds_arg1 The bounds of the first input used in producing the lower bound of the result.
         * @param ub_bounds_arg1 The bounds of the first input used in producing the upper bound of the result.
         * @param a1deps The dependencies of the first input on the target argument.
         * @param lb_bounds_arg2 The bounds of the second input used in producing the lower bound of the result.
         * @param ub_bounds_arg2 The bounds of the second input used in producing the upper bound of the result.
         * @param a2deps The dependencies of the second input on the target argument.
         * @return The resulting bound dependencies.
         */
        constexpr static BoundDependencies
            computation_uses(BoundID lb_bounds_arg1, BoundID ub_bounds_arg1, BoundDependencies a1deps,
                             BoundID lb_bounds_arg2, BoundID ub_bounds_arg2, BoundDependencies a2deps) noexcept
        {
            return BoundDependencies{
                depends_on_lb(lb_bounds_arg1, a1deps) || depends_on_lb(lb_bounds_arg2, a2deps),
                depends_on_ub(lb_bounds_arg1, a1deps) || depends_on_ub(lb_bounds_arg2, a2deps),
                depends_on_lb(ub_bounds_arg1, a1deps) || depends_on_lb(ub_bounds_arg2, a2deps),
                depends_on_ub(ub_bounds_arg1, a1deps) || depends_on_ub(ub_bounds_arg2, a2deps)
            };
        }
    };

    /**
     * @brief Output bound dependencies.
     * @param o
     * @param b
     * @return The output stream.
     */
    inline std::ostream& operator<<(std::ostream& o, const BoundDependencies& b) {
        o << std::boolalpha << "lb -> lb: " << b.lb_depends_on_lb << ", "
          << "lb -> ub: " << b.lb_depends_on_ub << ", "
          << "ub -> lb: " << b.ub_depends_on_lb << ", "
          << "ub -> ub: " << b.ub_depends_on_ub;
        return o;
    }

    /**
     * Compute the BoundDependencies for a given, bounded expression or predicate and a given argument index.
     *
     * @tparam BoundedMathExprOrPred
     * @tparam ArgIndex
     * @return The BoundDependencies for the given expression or predicate.
     */
    template<typename BoundedMathExprOrPred, std::size_t ArgIndex>
        static inline constexpr BoundDependencies compute_bound_dependencies() noexcept;
}

// Include implementation headers.
#include "bound_dependency_analysis/cbd_simple_deps.hpp"
#include "bound_dependency_analysis/cbd_meta_eval_tag.hpp"
#include "bound_dependency_analysis/cbd_bounded.hpp"
#include "bound_dependency_analysis/cbd_constants.hpp"
#include "bound_dependency_analysis/cbd_args.hpp"
#include "bound_dependency_analysis/cbd_arithmetic_ops.hpp"
#include "bound_dependency_analysis/compute_bound_dependencies.hpp"
