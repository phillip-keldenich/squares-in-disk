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

#include <vector>
#include <chrono>

#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"
#include "ivarp/constraint_system.hpp"
#include "ivarp/refactor_constraint_system.hpp"
#include "ivarp/prover_input/process_var_splitting.hpp"
#include "ivarp/cuda_transformer.hpp"

namespace ivarp {
    /**
     * @brief A struct wrapping the input arguments, constraints and variable splitting information passed to a prover.
     * Do not manually create this; use the function prover_input instead.
     *
     * @tparam NumArgs
     * @tparam VariableIndices_
     * @tparam VariableSplitting_
     * @tparam Context_
     * @tparam RuntimeBoundTable_
     * @tparam RuntimeConstraintTable_
     * @tparam CTArgBounds_
     */
    template<std::size_t NumArgs, typename VariableIndices_, typename VariableSplitting_,
             typename Context_, typename RuntimeBoundTable_, typename RuntimeConstraintTable_, typename CTArgBounds_>
        struct ProverInput
    {
    public:
        using VariableIndices = VariableIndices_;

    private:
        using ProcessVS = impl::ProcessVarSplitting<VariableSplitting_>;
        using VariableSplittingSequence_ = typename ProcessVS::ProcessedSequence;

    public:
        using VariableSplitInfo = impl::MakeSplitInfoSequence<VariableIndices, VariableSplittingSequence_>;
        using DynamicSplitInfo = impl::DynamicSplitInfos<VariableSplitInfo>;
        using StaticSplitInfo = impl::StaticSplitInfos<VariableSplitInfo>;
        using Context = Context_;
        using RuntimeBoundTable = RuntimeBoundTable_;
        using RuntimeConstraintTable = RuntimeConstraintTable_;
        using CTArgBounds = CTArgBounds_;
        using NumberType = typename Context::NumberType;
        static constexpr std::size_t num_args = NumArgs;
        static constexpr std::size_t num_vars = VariableIndices::size;
        static constexpr std::size_t initial_queue_size = ProcessVS::initial_queue_size;

        IVARP_DEFAULT_CM(ProverInput);

        static const auto& all_constraints(const RuntimeConstraintTable& rct) {
            return ivarp::template get<NumArgs-1>(rct);
        }

        RuntimeBoundTable runtime_bounds;
        RuntimeConstraintTable runtime_constraints;
        Array<NumberType, NumArgs> initial_runtime_bounds;
    };

    /**
     * Print information about a prover input (including bounds, constraints, etc.)
     * @tparam PIType
     * @param output The stream to print to.
     * @param prover_in The ProverInput to print.
     * @param fprinter A FunctionPrinter that is used to print functions/bounds/predicates.
     */
    template<typename PIType>
    static inline void print_prover_input(std::ostream& output, const PIType& prover_in, FunctionPrinter& fprinter);

    namespace impl {
        /**
         * Implementation of #prover_input.
         * @tparam Context
         * @tparam Enabler Always void. Do not touch; used for SFINAE only.
         */
        template<typename Context, typename Enabler = void> struct MakeProverInputImpl;
    }

    /**
     * @brief Create a prover input from a constraint system, a context and a variable splitting parameter.
     *
     * In particular, this function runs constraint system refactoring and
     * transformation to CUDA-compatible math_fns (if the Context::NumberType is IFloat or IDouble).
     *
     * @tparam Context
     * @tparam VariableSplitting
     * @tparam ConstraintSystemType
     * @param constraint_system
     * @return A ProverInput instance which can then be passed into a prover.
     */
    template<typename Context, typename VariableSplitting, typename ConstraintSystemType>
        static inline IVARP_H auto prover_input(const ConstraintSystemType& constraint_system)
    {
        return impl::MakeProverInputImpl<Context>::template prover_input<VariableSplitting>(constraint_system);
    }

    struct ProverSettings {
        /// The device IDs of CUDA devices to use. Default: use all available devices;
        /// if none are present, fall back to CPU. GPU use requires that run_prover is called from a .cu file,
        /// i.e., a file handled by nvcc instead of a regular C++ compiler.
        std::vector<int> cuda_device_ids = {-1};

        /// The number of generations, i.e., subdivisions applied to dynamic variables.
        unsigned generation_count = 8;

        /// The number of threads to use for CPU-based proofs. Default: Corresponding to the number of hardware threads.
        int thread_count = -1;

        /// The number of elements to take from the queue at once; default: Depending on what hardware is used.
        /// Determines the number of blocks for CUDA-based proofs.
        int dequeue_buffer_size = -1;

        /// The number of threads per block to execute, for CUDA-based proofs.
        int cuda_threads_per_block = 256;

        /// How many milliseconds to wait before beginning calls to the progress observer.
        std::chrono::milliseconds start_progress_after{8000};

        /// How many milliseconds to wait between two calls to the progress observer.
        std::chrono::milliseconds progress_every{2000};

        /// Limit on the number of iterations at a single node. Default depends on the prover core.
        int max_iterations_per_node = -1;

        /// Fraction of the total volume that a cuboid has to be reduced to in order to iterate at the same depth.
        /// Default depends on the prover core. 0 disables the criterion.
        float iteration_volume_criterion = -1.f;

        /// Fraction of the volume that a single dimension has to be reduced to in order to iterate at the same depth.
        /// Default depends on the prover core. 0 disables the criterion.
        float iteration_single_dimension_criterion = -1.f;
    };
}

#include "prover_input/make_prover_input_impl.hpp"
#include "prover_input/print_prover_input_impl.hpp"
