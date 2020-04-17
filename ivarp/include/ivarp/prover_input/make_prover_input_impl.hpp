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
// Created by Phillip Keldenich on 07.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    /**
     * Check whether the given Context has NumberType IFloat or IDouble.
     */
    template<typename Context> using IsCUDAProofContext = std::integral_constant<bool,
        std::is_same<typename Context::NumberType, IFloat>::value ||
        std::is_same<typename Context::NumberType, IDouble>::value
    >;

    /**
     * Transform a tuple of constraints by invoking transform_for_cuda on all elements.
     * @tparam ConstraintTuple
     * @tparam Indices
     * @param constraints
     * @return
     */
    template<typename ConstraintTuple, std::size_t... Indices> static inline IVARP_H auto
        transform_tuple_for_cuda(const ConstraintTuple& constraints, IndexPack<Indices...>)
    {
        return ivarp::make_tuple(transform_for_cuda(ivarp::template get<Indices>(constraints))...);
    }

    /**
     * Actually perform the transformation from a ConstraintSystem (tuple of constraints) to a ProverInput.
     * @tparam Context
     * @tparam VariableSplitting
     * @tparam ConstraintSystemType
     * @tparam CArgs
     * @param constraints
     * @return
     */
    template<typename Context, typename VariableSplitting, typename ConstraintSystemType, typename... CArgs>
        static inline IVARP_H auto actual_make_prover_input_impl(Tuple<CArgs...>&& constraints)
    {
        // check variable splitting
        static_assert(VariableSplitting::size == ConstraintSystemType::num_vars,
                      "Wrong number of elements in VariableSplitting!");

        using VariableIndices = typename ConstraintSystemType::VariableIndices;
        using InitialCTB = typename ConstraintSystemType::CTBounds;

        auto refactored = refactor_constraints<InitialCTB, Context>(ivarp::move(constraints));
        using RuntimeBoundTable = typename decltype(refactored)::template At<0>;
        using RuntimeConstraintTable = typename decltype(refactored)::template At<1>;
        using CTArgBounds = typename decltype(refactored)::template At<3>;

        return ProverInput<ConstraintSystemType::num_args, VariableIndices, VariableSplitting, Context,
                           RuntimeBoundTable, RuntimeConstraintTable, CTArgBounds>
        {
            ivarp::template get<0>(ivarp::move(refactored)),
            ivarp::template get<1>(ivarp::move(refactored)),
            ivarp::template get<2>(ivarp::move(refactored))
        };
    }

    /**
     * Implementation of MakeProverInputImpl for CUDA-compatible contexts.
     * @tparam Context
     */
    template<typename Context> struct MakeProverInputImpl<Context, std::enable_if_t<IsCUDAProofContext<Context>::value>> {
        template<typename VariableSplitting, typename ConstraintSystemType> static IVARP_H auto
            prover_input(const ConstraintSystemType& constraint_system)
        {
            // transform to CUDA compatible constraints
            using CSConstraintInds = TupleIndexPack<typename ConstraintSystemType::Constraints>;
            auto cuda_constraints = transform_tuple_for_cuda(constraint_system.constraints(), CSConstraintInds{});
            return actual_make_prover_input_impl<Context, VariableSplitting, ConstraintSystemType>(
                ivarp::move(cuda_constraints)
            );
        }
    };

    /**
     * Implementation of MakeProverInputImpl for non-CUDA-compatible contexts, such as contexts using rational numbers.
     * @tparam Context
     */
    template<typename Context> struct MakeProverInputImpl<Context, std::enable_if_t<!IsCUDAProofContext<Context>::value>> {
        template<typename VariableSplitting, typename ConstraintSystemType> static IVARP_H auto
            prover_input(const ConstraintSystemType& constraint_system)
        {
            auto cs = constraint_system.constraints();
            return actual_make_prover_input_impl<Context, VariableSplitting, ConstraintSystemType>(ivarp::move(cs));
        }
    };
}
}
