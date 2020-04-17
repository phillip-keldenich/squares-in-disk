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
// Created by Phillip Keldenich on 26.11.19.

#pragma once

namespace ivarp {
namespace impl {
    template<typename Runner_> class ProverThread {
    public:
        using Runner = Runner_;
        using Prover = typename Runner::Prover;
        using ProverSettings = typename Runner::ProverSettings;
        using Variables = typename Prover::Variables;
        using Constraints = typename Prover::Constraints;
        using OnCritical = typename Runner::OnCritical;
        using NumberType = typename ProverSettings::NumberType;
        using DynamicEntryType = typename Runner::DynamicEntryType;
        using StaticIndices = typename Runner::StaticIndices;
        using Context = typename Runner::Context;
        template<std::size_t I> using VarIndex = typename Runner::template VarIndex<I>;

        explicit ProverThread(Runner* runner, std::size_t index) :
            runner(runner),
            thread_index(index),
            thread{}
        {}

        void join() {
            if(thread.joinable()) {
                thread.join();
            }
        }

        void start(bool only_init) {
            thread = std::thread(&ProverThread::entry_point, this, only_init);
        }

        std::size_t cuboids() const noexcept {
            return num_cuboids;
        }

    private:
        /// The entry point for the prover threads (also used during initialization).
        void entry_point(bool only_init) noexcept {
            SetRoundDown round_down;
            if(!run_init()) {
                return;
            }

            if(!only_init && Runner::use_cpu_threads) {
                // busy-wait until all threads are done initializing
                ++runner->threads_done_with_init;
                while(runner->threads_done_with_init < runner->num_threads) {
                    std::this_thread::yield();
                }
                main();
            }
        }

        /// Run the initialization, i.e., fill the initial dynamically-split cuboid queue.
        bool run_init() {
            DynamicEntryType root_entry;
            if(runner->constraint_prop.template compute_new_var_bounds<0, Context>(root_entry.bounds).empty) {
                if(thread_index == 0) {
                    std::cerr << "Warning: The range for the first variable is empty - no feasible cuboids!"
                              << std::endl;
                    runner->queue.clear();
                }
                return false;
            }

            Splitter<NumberType> split_v0(root_entry.bounds[0], static_cast<int>(Runner::first_var_subintervals));
            for(std::size_t i = thread_index; i < Runner::first_var_subintervals; i += runner->num_threads) {
                std::size_t offset = i * (Variables::initial_dynamic_cuboids / Runner::first_var_subintervals);
                init_dynamic_vars(split_v0.subrange(static_cast<int>(i)), offset);
            }

            return true;
        }

        /// The main loop of the prover threads; only run if we are using the CPU for the main phase.
        void main() {
            DynamicEntryType entry;
            while(runner->queue.pop_into(entry)) {
                if(!entry.empty) {
                    DynamicEntryType criticals_in_entry;
                    handle_entry(entry, &criticals_in_entry);
                }
            }
        }

        /// Handle a single cuboid queue entry.
        void handle_entry(DynamicEntryType& entry, DynamicEntryType* criticals_in_entry) {
            for(std::size_t at_this_depth = 0; true; ++at_this_depth) {
                // actually work on the non-dynamic variables
                criticals_in_entry->empty = true;
                non_dynamic_phase(entry, criticals_in_entry, StaticIndices{});
                if(criticals_in_entry->empty) {
                    ++num_cuboids;
                    return; // no criticals: we are done with this entry.
                }

                // otherwise, check whether we reduced the critical ranges enough to iterate on the same entry.
                bool iterate = runner->iterate_on_entry(entry, criticals_in_entry, at_this_depth);
                runner->restrict_non_dynamic_bounds(entry, criticals_in_entry);
                if(iterate) {
                    ++num_cuboids;
                    continue; // re-run this entry with the new reduced critical space
                }

                // either split the dynamic variables or report critical cuboids
                if(entry.depth < ProverSettings::max_dynamic_split_depth) {
                    runner->split_dynamic_vars(entry);
                } else {
                    non_dynamic_phase_report_criticals(entry, StaticIndices{});
                }
                return; // either way, we are done with this entry
            }
        }

        /// Initialize the queue of dynamic variables.
        void init_dynamic_vars(NumberType v0range, std::size_t queue_offset) {
            DynamicEntryType entry;
            entry.bounds[0] = v0range;

            // make the compiler happy with some initialization for the other variables (bounds will be overwritten)
            for(std::size_t i = 1; i < Variables::num_vars; ++i) {
                entry.bounds[i] = NumberType{0};
            }

            init_dynamic_var(entry, queue_offset, VarIndex<0>{});
        }

        /// Initialize the dynamic variables, beginning with CurVar.
        template<std::size_t CurVar> std::enable_if_t<(CurVar < Variables::num_dynamic_vars-1)>
            init_dynamic_var(DynamicEntryType entry, std::size_t& queue_offset,
                             std::integral_constant<std::size_t, CurVar>)
        {
            if(runner->constraint_prop.template compute_new_var_bounds<CurVar+1, Context>(entry.bounds).empty) {
                ++num_cuboids;
                init_dynamic_var_empty(queue_offset, VarIndex<CurVar+1>{});
                return;
            }

            constexpr std::size_t applics = ProverSettings::dynamic_phase_propagation_applications_per_constraint;
            constexpr auto num_subs = static_cast<int>(Variables::template NumSubdivisionsOf<CurVar+1>::value);
            Splitter<NumberType> split_vnext(get<CurVar+1>(entry.bounds), num_subs);
            for(int i = 0; i < num_subs; ++i) {
                DynamicEntryType sub_entry = entry;
                get<CurVar+1>(sub_entry.bounds) = split_vnext.subrange(i);
                if(runner->constraint_prop.template dynamic_post_split_var<CurVar+1, applics, Context>(sub_entry.bounds).empty)
                {
                    ++num_cuboids;
                    init_dynamic_var_empty(queue_offset, VarIndex<CurVar+2>{});
                } else {
                    init_dynamic_var(sub_entry, queue_offset, VarIndex<CurVar+1>{});
                }
            }
        }

        /// Init the dynamic variables; this is the function used for the last dynamic variable.
        void init_dynamic_var(DynamicEntryType entry, std::size_t& queue_offset,
                              VarIndex<Variables::num_dynamic_vars-1>)
        {
            for(std::size_t i = Variables::num_dynamic_vars; i < Variables::num_vars; ++i)  {
                entry.bounds[i].set_lb(-infinity);
                entry.bounds[i].set_ub(infinity);
            }
            runner->queue[queue_offset++] = entry;
        }

        /// The range for variable EmptyVar is empty during initialization;
        /// set the corresponding queue entries to empty.
        template<std::size_t EmptyVar> std::enable_if_t<(EmptyVar < Variables::num_dynamic_vars)>
            init_dynamic_var_empty(std::size_t& queue_offset,
                                   std::integral_constant<std::size_t, EmptyVar>) noexcept
        {
            constexpr auto num_subs = static_cast<int>(Variables::template NumSubdivisionsOf<EmptyVar>::value);
            for(int i = 0; i < num_subs; ++i) {
                init_dynamic_var_empty(queue_offset, VarIndex<EmptyVar+1>{});
            }
        }
        void init_dynamic_var_empty(std::size_t& queue_offset,
                                    VarIndex<Variables::num_dynamic_vars>) noexcept
        {
            runner->queue[queue_offset++].empty = true;
        }

        /// Run the static phase; next, split the variable indicated by index S1.
        template<typename CriticalHandler, std::size_t S1, std::size_t... Statics>
            inline void static_phase(DynamicEntryType entry, const CriticalHandler& handler,
                                     IndexPack<S1,Statics...>)
        {
            constexpr std::size_t apps = ProverSettings::static_phase_propagation_applications_per_constraint;
            constexpr int subs = Variables::template NumSubdivisionsOf<S1>::value;
            if(runner->constraint_prop.template restrict_new_var_bounds<S1,Context>(entry.bounds).empty) {
                ++num_cuboids;
                return;
            }
            Splitter<NumberType> s1split(entry.bounds[S1], subs);
            for(int si = 0; si < subs; ++si) {
                entry.bounds[S1] = s1split.subrange(si);
                if(runner->constraint_prop.template static_post_split_var<S1, apps, Context>(entry.bounds).empty) {
                    ++num_cuboids;
                    continue;
                }
                static_phase(entry, handler, IndexPack<Statics...>{});
            }
        }

        /// We have worked our way through all variables; if the constraints cannot prune the entry, report it to the
        /// critical handler.
        template<typename CriticalHandler> inline void
            static_phase(const DynamicEntryType& entry, const CriticalHandler& handler, IndexPack<>)
        {
            ++num_cuboids;
            if(!runner->constraint_prop.template can_constraints_prune<Variables::num_vars, Context>(entry.bounds)) {
                handler(entry.bounds);
            }
        }

        /// Run the static phase, collecting (a superset of) the union of all criticals in criticals_in_entry.
        template<std::size_t... Statics> inline void
            non_dynamic_phase(const DynamicEntryType& entry, DynamicEntryType* criticals_in_entry,
                              IndexPack<Statics...> s)
        {
            const auto add_critical_to_ptr = [criticals_in_entry] (const auto& args) {
                if(criticals_in_entry->empty) {
                    criticals_in_entry->empty = false;
                    for(std::size_t i = 0; i < Variables::num_vars; ++i) {
                        criticals_in_entry->bounds[i] = args[i];
                    }
                } else {
                    for(std::size_t i = 0; i < Variables::num_vars; ++i) {
                        criticals_in_entry->bounds[i].do_join(args[i]);
                    }
                }
            };
            static_phase(entry, add_critical_to_ptr, s);
        }

        /// Run the static phase, reporting all criticals to the OnCritical handler.
        template<std::size_t... Statics> inline void
            non_dynamic_phase_report_criticals(const DynamicEntryType& e, IndexPack<Statics...> s)
        {
            const auto reporter = [&] (const auto& args) {
                ++runner->reported_criticals;
                (*(runner->report_criticals))(Context{}, args);
            };
            static_phase(e, reporter, s);
        }

        Runner* const runner;
        const std::size_t thread_index;
        std::size_t num_cuboids{0};
        std::thread thread;
    };
}
}
