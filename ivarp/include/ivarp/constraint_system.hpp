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
// Created by Phillip Keldenich on 16.01.20.
//

#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#include "ivarp/variable_description.hpp"
#include "ivarp/constraint_system/extract_ct_bounds.hpp"
#include "ivarp/constraint_system/extract_bound_constraints.hpp"

namespace ivarp {
    template<typename Constraints_, typename CTBounds_, typename VarIndexPack_>
        class ConstraintSystem : public ArgNameLookup
    {
    public:
        using Constraints = Constraints_;
        using CTBounds = CTBounds_;
        using VariableIndices = VarIndexPack_;

        template<typename ArgDescriptionsTuple, std::size_t... ArgDIndex> explicit IVARP_H
        ConstraintSystem(Constraints&& constraints, ArgDescriptionsTuple&& arg_desc, IndexPack<ArgDIndex...>) noexcept :
            m_constraints(ivarp::move(constraints))
        {
            ConstructWithAny{
                (m_arg_names[ArgDescriptionsTuple::template At<ArgDIndex>::index] =
                        ivarp::move(ivarp::template get<ArgDIndex>(arg_desc).name))...
            };
        }

        constexpr static std::size_t num_args = TupleSize<CTBounds>::value;
        constexpr static std::size_t num_vars = VariableIndices::size;

        IVARP_H std::string arg_name(std::size_t arg_index) const override {
            if(arg_index >= num_args) {
                throw std::out_of_range("arg_index out of range in ConstraintSystem::arg_name!");
            }

            return m_arg_names[arg_index];
        }

        const Constraints& constraints() const noexcept {
            return m_constraints;
        }

    private:
        Constraints m_constraints;
        std::string m_arg_names[num_args];
    };

    /**
     * Create a constraint system from variable descriptions, value descriptions and constraints.
     *
     * @tparam VarsValsConstrs
     * @param vvc
     * @return
     */
    template<typename... VarsValsConstrs> static inline IVARP_H auto constraint_system(VarsValsConstrs&&... vvc) {
        auto all_params = ivarp::make_tuple(ivarp::forward<VarsValsConstrs>(vvc)...);
        using ArgDescriptionIndices = FilteredIndexPackType<IsArgDescription, VarsValsConstrs...>;
        using ConstraintIndices = FilteredIndexPackType<Not<IsArgDescription>::template Predicate, VarsValsConstrs...>;

        auto args = filter_tuple(ivarp::move(all_params), ArgDescriptionIndices{});
        using VariableArgs = impl::VariableArgIndices<decltype(args)>;
        using CTArgBounds = typename impl::ExtractCTBounds<decltype(args)>::Bounds;
        auto bound_constrs = impl::extract_bound_constraints(args);
        auto constrs = concat_tuples(bound_constrs, filter_tuple(ivarp::move(all_params), ConstraintIndices{}));
        return ConstraintSystem<decltype(constrs), CTArgBounds, VariableArgs>{
            ivarp::move(constrs), ivarp::move(args), TupleIndexPack<decltype(args)>{}
        };
    }
}
