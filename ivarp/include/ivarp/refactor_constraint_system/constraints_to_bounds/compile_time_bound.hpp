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
    template<typename BoundFunction, BoundDirection Direction> struct CompileTimeBound {
        static constexpr bool success = true;
        static constexpr bool runtime_success = true;
        static constexpr BoundDirection direction = Direction;
        using Bound = BoundFunction;

        IVARP_DEFAULT_CM(CompileTimeBound);

        explicit CompileTimeBound(Bound&& b) noexcept : bound(ivarp::move(b)) {}
        explicit CompileTimeBound(const Bound& b) : bound(b) {}

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(typename Context, std::size_t TargetArg, typename BoundArr),
            typename Context::NumberType,
            BoundID apply_to(BoundArr& args) const {
                auto interval = bound.template array_evaluate<Context>(args);
                if(fixed_point_bounds::possibly_undefined<Bound>(interval)) {
                    return BoundID::NONE;
                }
                auto& target = args[TargetArg];
                if(Direction == BoundDirection::LEQ) {
                    // TargetArg <= Bound(args), i.e., ub(Bound(args)) is an upper bound on target
                    if(interval.ub() < target.ub()) {
                        target.set_ub(interval.ub());
                        return interval.ub() < target.lb() ? BoundID::EMPTY : BoundID::UB;
                    }
                } else if(Direction == BoundDirection::GEQ) {
                    // TargetArg >= Bound(args), i.e., lb(Bound(args)) is a lower bound on target
                    if(interval.lb() > target.lb()) {
                        target.set_lb(interval.lb());
                        return interval.lb() > target.ub() ? BoundID::EMPTY : BoundID::LB;
                    }
                } else {
                    // Target == Bound(args)
                    BoundID lbchanged = BoundID::NONE;
                    BoundID ubchanged = BoundID::NONE;
                    if(interval.ub() < target.ub()) {
                        target.set_ub(interval.ub());
                        ubchanged = BoundID::UB;
                    }
                    if(interval.lb() > target.lb()) {
                        target.set_lb(interval.lb());
                        lbchanged = BoundID::LB;
                    }
                    if(target.ub() < target.lb()) {
                        return BoundID::EMPTY;
                    } else {
                        return static_cast<BoundID>(static_cast<unsigned>(lbchanged) |
                                                    static_cast<unsigned>(ubchanged));
                    }
                }

                return BoundID::NONE;
            }
        )

        BoundDirection get_direction() const noexcept {
            return direction;
        }

        Bound bound;
    };

    template<BoundDirection B> struct NegateCTBImpl {
    private:
        static constexpr BoundDirection result_direction =
            (B == BoundDirection::GEQ) ? BoundDirection::LEQ :
            (B == BoundDirection::LEQ) ? BoundDirection::GEQ :
                                         BoundDirection::NONE;

        struct Failure {
            template<typename CTB>
                static inline auto negate(CTB&&) noexcept
            {
                return RewriteFailed{};
            }
        };

        struct Success {
            template<typename CTB>
                static inline auto negate(CTB&& c)
            {
                using ResultType = CompileTimeBound<typename CTB::Bound, result_direction>;
                return ResultType{forward_other<CTB>(c.bound)};
            }
        };

        using LazySF = std::conditional_t<result_direction == BoundDirection::NONE, Failure, Success>;

    public:
        template<typename CTB>
            static inline auto negate(CTB&& c)
        {
            return LazySF::negate(ivarp::forward<CTB>(c));
        }
    };

    template<typename BoundFunction, BoundDirection Direction>
        static inline auto negate_bound(const CompileTimeBound<BoundFunction,Direction>& b)
    {
        return NegateCTBImpl<Direction>::negate(b);
    }

    template<typename BoundFunction, BoundDirection Direction>
        static inline auto negate_bound(CompileTimeBound<BoundFunction,Direction>&& b) noexcept
    {
        return NegateCTBImpl<Direction>::negate(ivarp::move(b));
    }

    template<typename ArgBounds, typename BoundsSoFar, typename... CTBs> struct IntersectCTBsImpl {
        using Bounds = BoundsSoFar;
    };

    template<typename ArgBounds, typename BoundsSoFar, typename CTB1, typename... CTBs>
        struct IntersectCTBsImpl<ArgBounds, BoundsSoFar, CTB1, CTBs...>
    {
    private:
        using CTB1Result = CompileTimeBounds<typename CTB1::Bound, ArgBounds>;
        static constexpr std::int64_t new_lb = ((CTB1::direction & BoundDirection::GEQ) != BoundDirection::NONE) ?
                (ivarp::max)(BoundsSoFar::lb, CTB1Result::lb) : BoundsSoFar::lb;
        static constexpr std::int64_t new_ub = ((CTB1::direction & BoundDirection::LEQ) != BoundDirection::NONE) ?
                (ivarp::min)(BoundsSoFar::ub, CTB1Result::ub) : BoundsSoFar::ub;

    public:
        using Bounds = typename IntersectCTBsImpl<ArgBounds, ExpressionBounds<new_lb,new_ub>, CTBs...>::Bounds;
    };

    template<std::size_t TargetArg, typename ArgBounds, typename... CTBs> struct IntersectCTBs {
        using Bounds = typename IntersectCTBsImpl<ArgBounds, typename ArgBounds::template At<TargetArg>,
                                                  CTBs...>::Bounds;
        static_assert(Bounds::lb <= Bounds::ub,
                      "Empty bounds resulting from intersection; proof successful at compile time?!");
    };

    template<std::size_t TargetArg, typename ArgBounds, typename CTBTuple> struct IntersectCTBTuple;
    template<std::size_t TargetArg, typename ArgBounds, typename... CTBs>
        struct IntersectCTBTuple<TargetArg, ArgBounds, Tuple<CTBs...>> :
            IntersectCTBs<TargetArg, ArgBounds, CTBs...>
    {};
}
}
