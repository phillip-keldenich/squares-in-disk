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
// Created by Phillip Keldenich on 06.02.20.
//

#pragma once

#include "ivarp/metaprogramming.hpp"
#include "ivarp/compile_time_bounds.hpp"
#include "ivarp/variable_description.hpp"

namespace ivarp {
namespace impl {
    template<typename ArgDescriptionTuple, std::size_t ArgIndex> struct ExtractArgDescriptionFor {
    private:
        template<std::size_t TupIndex> using IsArgIndex = std::integral_constant<bool,
            ArgDescriptionTuple::template At<TupIndex>::index == ArgIndex
        >;

        using Descriptions = FilterIndexPack<IsArgIndex, TupleIndexPack<ArgDescriptionTuple>>;
        static_assert(Descriptions::size == 1, "There must be exactly one description for each argument index!");

    public:
        using Type = typename ArgDescriptionTuple::template At<Descriptions::template At<0>::value>;
    };

    template<typename ArgDescriptionTuple> struct ArgIndexIsVariableNotValue {
        template<std::size_t ArgIndex> struct Predicate {
        private:
            using Description = typename ExtractArgDescriptionFor<ArgDescriptionTuple, ArgIndex>::Type;

        public:
            static constexpr bool value = ivarp::IsVariableDescription<Description>::value;
        };
    };

    template<typename ArgDescriptionTuple> struct VariableArgIndicesImpl {
        using Type = FilterIndexPack<ArgIndexIsVariableNotValue<ArgDescriptionTuple>::template Predicate,
                                     TupleIndexPack<ArgDescriptionTuple>>;
    };
    template<typename ArgDescriptionTuple> using VariableArgIndices =
        typename VariableArgIndicesImpl<ArgDescriptionTuple>::Type;

    template<typename ArgDescriptionTuple, typename BoundsSoFar, typename Indices> struct ExtractCTBoundsImpl;
    template<typename ArgDescriptionTuple, typename BoundsSoFar, std::size_t I1, std::size_t... RemInds>
        struct ExtractCTBoundsImpl<ArgDescriptionTuple, BoundsSoFar, IndexPack<I1,RemInds...>>
    {
        using Description = typename ExtractArgDescriptionFor<ArgDescriptionTuple, I1>::Type;
        using LB = typename Description::LBType;
        using UB = typename Description::UBType;
        static constexpr std::int64_t lb = CompileTimeBounds<LB, BoundsSoFar>::lb;
        static constexpr std::int64_t ub = CompileTimeBounds<UB, BoundsSoFar>::ub;
        using NewBounds = typename BoundsSoFar::template Append<ExpressionBounds<lb,ub>>;
        using Bounds = typename ExtractCTBoundsImpl<ArgDescriptionTuple, NewBounds, IndexPack<RemInds...>>::Bounds;
    };
    template<typename ArgDescriptionTuple, typename BoundsSoFar>
        struct ExtractCTBoundsImpl<ArgDescriptionTuple, BoundsSoFar, IndexPack<>>
    {
        using Bounds = BoundsSoFar;
    };

    template<typename ArgDescriptionTuple> struct ExtractCTBounds :
        ExtractCTBoundsImpl<ArgDescriptionTuple, Tuple<>, TupleIndexPack<ArgDescriptionTuple>>
    {};
}
}
