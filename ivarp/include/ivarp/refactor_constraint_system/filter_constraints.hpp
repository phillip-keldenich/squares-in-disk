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

/**
 * @file filter_constraints.hpp
 * Implements a metafunction that removes always-true constraints from a bounded constraint tuple.
 */

namespace ivarp {
    template<typename BoundedConstraintTuple> struct FilterConstraints {
    private:
        template<std::size_t Index> struct NotAlwaysTrue {
            using TAt = typename BoundedConstraintTuple::template At<Index>;
            static constexpr bool value = !TAt::lb;
        };

        template<typename BCT, std::size_t... Inds> static inline auto do_filter(BCT&& constraints, IndexPack<Inds...>)
        {
            return ivarp::make_tuple(ivarp::template get<Inds>(ivarp::forward<BCT>(constraints))...);
        }

    public:
        template<typename BCT, typename Pack> static inline auto filter(BCT&& constraints, Pack) {
            using FilteredPack = FilterIndexPack<NotAlwaysTrue, Pack>;
            return do_filter(ivarp::forward<BCT>(constraints), FilteredPack{});
        }

        using Type = decltype(filter(std::declval<BoundedConstraintTuple>(), TupleIndexPack<BoundedConstraintTuple>{}));
    };

    template<typename BoundedConstraintTuple>
        static inline auto filter_constraints(BoundedConstraintTuple&& t)
    {
        using TT = BareType<BoundedConstraintTuple>;
        return FilterConstraints<TT>::filter(ivarp::forward<BoundedConstraintTuple>(t), TupleIndexPack<TT>{});
    }
}
