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
// Created by Phillip Keldenich on 07.11.19.
//

#pragma once

#include <thread>

namespace ivarp {
    enum class NonDynamicPhaseMethod {
        CPU, CUDA
    };

    template<typename NumberType_> struct DefaultProverSettingsWithNumberType {
        static constexpr std::size_t dynamic_phase_propagation_applications_per_constraint = 5;
        static constexpr std::size_t static_phase_propagation_applications_per_constraint = 2;
        static constexpr std::size_t max_dynamic_split_depth = 12;
        static constexpr std::size_t max_iterations_per_depth = 4;
        static constexpr std::size_t initial_progress_wait_msec = 10000;
        static constexpr std::size_t progress_interval_msec = 2000;
        static constexpr int subintervals_per_dynamic_var_split = 4;

        // only used if NumberType is IRational
        static constexpr std::size_t irrational_precision = default_irrational_precision;

        static std::size_t cpu_threads() noexcept {
            return std::thread::hardware_concurrency();
        }

        static constexpr NonDynamicPhaseMethod non_dynamic_phase_method = NonDynamicPhaseMethod::CPU;
        static constexpr CuboidQueueOrder cuboid_queue_order = CuboidQueueOrder::LIFO;
        using NumberType = NumberType_;
    };

    using DefaultProverSettings = DefaultProverSettingsWithNumberType<IDouble>;

    template<typename Vars, typename Constrs> struct Prover {
        using Variables = Vars;
        using Constraints = Constrs;

        static_assert(IsVariables<Vars>::value, "Prover not passed Variables!");
        static_assert(IsConstraints<Constrs>::value, "Prover not passed Constraints!");

        explicit Prover(const Variables& vars, const Constraints& constrs) :
            vars(vars), var_names(get_var_names(vars)),
            constraints(constrs)
        {}

        template<typename ProverSettings = DefaultProverSettings,
                 typename ProgressObserver = DefaultProgressObserver,
                 typename OnCritical>
            inline bool run(const OnCritical& c, ProofInformation* information = nullptr,
                            ProgressObserver progress = ProgressObserver{}) const;

        const std::string& variable_name(std::size_t index) const noexcept {
            assert(index < Variables::num_vars);
            return var_names[index];
        }

        Variables vars;
        std::array<std::string,Variables::num_vars> var_names;
        Constraints constraints;

        static constexpr std::size_t num_dynamic_vars = Variables::num_dynamic_vars;
        static constexpr std::size_t num_vars = Variables::num_vars;
        static_assert(num_dynamic_vars > 0, "Currently, we need at least one dynamic variable!");

        template<typename ProverSettings = DefaultProverSettings>
            inline std::vector<impl::DynamicEntry<typename ProverSettings::NumberType, num_vars>>
                generate_dynamic_phase_queue() const;

    private:
        /// Create an array of variable names.
        static std::array<std::string,Variables::num_vars> get_var_names(const Variables& v) {
            std::array<std::string,Variables::num_vars> result;
            get_var_names(v, result, IndexRange<0,Variables::num_vars>{});
            return result;
        }
        template<std::size_t I1, std::size_t... Inds>
            static void get_var_names(const Variables& v,
                                      std::array<std::string,Variables::num_vars>& out,
                                      IndexPack<I1,Inds...>)
        {
            out[I1] = get<I1>(v.vars).name();
            get_var_names(v, out, IndexPack<Inds...>{});
        }
        static void get_var_names(const Variables&, std::array<std::string,Variables::num_vars>&, IndexPack<>) {}
    };

    template<typename Vars, typename Constrs> static inline
        Prover<std::decay_t<Vars>, std::decay_t<Constrs>>
            prover(const Vars& v, const Constrs& c)
    {
        return Prover<std::decay_t<Vars>, std::decay_t<Constrs>>{v, c};
    }
}
