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
// Created by Phillip Keldenich on 04.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename ArgBounds, typename NewConstraints> struct RuntimeConstraintTableImpl {
    private:
        constexpr static std::size_t num_args = ArgBounds::size;

        template<typename Constraint, std::size_t ArgIndex>
        using IsUnrewrittenWRT = std::integral_constant<bool, !impl::SuccessfulCTRewrite<ArgIndex, ArgBounds, Constraint>::value>;

        template<typename Constraint> struct IsUnrewritten {
        private:
            template<std::size_t... Args> static constexpr bool is_unrewritten(IndexPack<Args...>) {
                return AllOf<IsUnrewrittenWRT<Constraint, Args>::value...>::value;
            }

        public:
            static constexpr bool value = is_unrewritten(IndexRange<0,num_args>{});
        };

        struct LastRow {
            template<std::size_t, std::size_t> using Lazy = std::true_type;
        };

        struct NotLastRow {
            struct NotRightIndex {
                template<typename Constraint> struct Lazy : std::false_type {};
            };

            struct RightIndex {
                template<typename Constraint> struct Lazy : IsUnrewritten<Constraint> {};
            };

            template<std::size_t CI, std::size_t RI> struct Lazy {
            private:
                using Constraint = typename NewConstraints::template At<CI>;
                static constexpr bool right_index = (NumArgs<Constraint>::value == RI+1);
                using CheckIndex = std::conditional_t<right_index, RightIndex, NotRightIndex>;
            public:
                static constexpr bool value = CheckIndex::template Lazy<Constraint>::value;
            };
        };

        template<std::size_t ConstraintIndex, std::size_t RowIndex> struct IncludeInRow :
            std::conditional_t<RowIndex == num_args-1, LastRow, NotLastRow>::template Lazy<ConstraintIndex, RowIndex>
        {};

        template<std::size_t RowIndex, typename ConstraintRange> struct ConstraintsInRow {
        private:
            template<std::size_t CI> using DoInclude = IncludeInRow<CI, RowIndex>;
        public:
            using Type = FilterIndexPack<DoInclude, ConstraintRange>;
        };

        template<std::size_t Row, std::size_t... CInds> static inline IVARP_H auto
            make_row(const NewConstraints& constraints, IndexPack<CInds...>)
        {
            return ivarp::make_tuple(ivarp::template get<CInds>(constraints)...);
        }

        template<std::size_t... Rows>
        static inline IVARP_H auto make(const NewConstraints& constraints, IndexPack<Rows...>)
        {
            using ConstraintRange = TupleIndexPack<NewConstraints>;
            return ivarp::make_tuple(
                make_row<Rows>(constraints, typename ConstraintsInRow<Rows, ConstraintRange>::Type{})...
            );
        }

    public:
        static inline IVARP_H auto make(const NewConstraints& constraints) {
            return make(constraints, IndexRange<0,num_args>{});
        }
    };
}

    /**
     * Create a tuple of tuples that maps arg indices to the set of constraints that should be checked
     * after the corresponding argument was split. The last entry contains all (except for filtered out constraints).
     *
     * @tparam VarAndValBounds
     * @tparam FilteredConstraints
     */
    template<typename VarAndValBounds, typename FilteredConstraints> static inline auto IVARP_H
        make_runtime_constraint_table(const FilteredConstraints& constraints)
    {
        return impl::RuntimeConstraintTableImpl<VarAndValBounds, FilteredConstraints>::make(constraints);
    }
}
