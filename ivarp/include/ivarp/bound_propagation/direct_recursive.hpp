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
// Created by Phillip Keldenich on 17.02.20.
//

#pragma once

namespace ivarp {
namespace propagate_direct {
namespace impl {
    template<std::size_t NumBounds> struct DirectRecursivePropInfo {
        static constexpr std::size_t num_bounds = NumBounds;

        DirectRecursivePropInfo() noexcept { // NOLINT
            clear();
        }

        void clear() noexcept {
            std::fill_n(+rec_count, num_bounds, std::uint8_t{0});
        }

        template<std::uint8_t RecLimit>
            bool recurse_on(std::size_t i) noexcept
        {
            if(rec_count[i] >= RecLimit) {
                return false;
            }
            rec_count[i] += 1;
            return true;
        }

        std::uint8_t rec_count[num_bounds];
    };

    template<std::uint8_t RecLimit, std::size_t BoundIndex, typename Context, typename RBT>
        static inline PropagationResult recursively_apply_bound(const RBT& runtime_bounds,
                                                                DirectRecursivePropInfo<RBT::num_bounds>& info,
                                                                typename Context::NumberType* apply_to);

    template<std::uint8_t RecLimit, typename Context, typename RBT>
        static inline PropagationResult recursively_apply_bounds(const RBT& runtime_bounds,
                                                                 DirectRecursivePropInfo<RBT::num_bounds>& info,
                                                                 typename Context::NumberType*,
                                                                 IndexPack<>)
    {
        return {false};
    }

    template<std::uint8_t RecLimit, typename Context, typename RBT, std::size_t B1, std::size_t... B>
        static inline PropagationResult recursively_apply_bounds(const RBT& runtime_bounds,
                                                                 DirectRecursivePropInfo<RBT::num_bounds>& info,
                                                                 typename Context::NumberType* apply_to,
                                                                 IndexPack<B1,B...>)
    {
        if(recursively_apply_bound<RecLimit, B1, Context>(runtime_bounds, info, apply_to).empty) {
            return {true};
        }
        return recursively_apply_bounds<RecLimit, Context>(runtime_bounds, info, apply_to, IndexPack<B...>{});
    }

    template<std::uint8_t RecLimit, std::size_t BoundIndex, typename Context, typename RBT>
        static inline PropagationResult recursively_apply_bound(const RBT& runtime_bounds,
                                                                DirectRecursivePropInfo<RBT::num_bounds>& info,
                                                                typename Context::NumberType* apply_to)
    {
        using ::ivarp::impl::BoundEvent;
        if(info.rec_count[BoundIndex] >= RecLimit) {
            return {false};
        }
        ++info.rec_count[BoundIndex];

        auto app_result = runtime_bounds.template apply_bound<BoundIndex,Context>(apply_to);
        constexpr std::size_t target = decltype(app_result)::target;
        using LBUpdate = typename RBT::template UpdateBounds<BoundEvent<target, BoundID::LB>>;
        using UBUpdate = typename RBT::template UpdateBounds<BoundEvent<target, BoundID::UB>>;
        using BothUpdate = typename RBT::template UpdateBounds<BoundEvent<target, BoundID::BOTH>>;
        switch(app_result.bound_id) {
            case BoundID::EMPTY:
                return {true};

            case BoundID::NONE:
                return {false};

            case BoundID::LB:
                return recursively_apply_bounds<RecLimit, Context>(runtime_bounds, info, apply_to, LBUpdate{});

            case BoundID::UB:
                return recursively_apply_bounds<RecLimit, Context>(runtime_bounds, info, apply_to, UBUpdate{});

            case BoundID::BOTH:
                return recursively_apply_bounds<RecLimit, Context>(runtime_bounds, info, apply_to, BothUpdate{});
        }
    }
}

    template<std::uint8_t RecLimit, typename BoundEv, typename Context, typename RBT>
        static inline PropagationResult propagate(const RBT& runtime_bounds, typename Context::NumberType* apply_to)
    {
        impl::DirectRecursivePropInfo<RBT::num_bounds> info;
        using Update = typename RBT::template UpdateBounds<BoundEv>;
        return {impl::recursively_apply_bounds<RecLimit, Context>(runtime_bounds, info, apply_to, Update{}).empty};
    }
}
}
