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
// Created by Phillip Keldenich on 13.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Forward-declare ProverThread; in this header, ProverThread is an incomplete type.
    template<typename Runner> class ProverThread;

    /// ProofRunner implements the running of the proof.
    template<typename Prover_, typename ProverSettings_, typename OnCritical_, typename ProgressObserver_>
    class ProofRunner
    {
    public:
        /// The prover settings.
        using ProverSettings = ProverSettings_;
        /// The prover type.
        using Prover = Prover_;
        /// The variables we work on.
        using Variables = typename Prover::Variables;
        /// The constraint collection we are working on.
        using Constraints = typename Prover::Constraints;
        /// The method we report critical intervals to. Is called with a Context object and an array of critical values.
        using OnCritical = OnCritical_;
        /// The progress observer type.
        using ProgressObserver = ProgressObserver_;
        /// The number type we work on.
        using NumberType = typename ProverSettings::NumberType;
        /// Entry in our cuboid queue.
        using DynamicEntryType = DynamicEntry<NumberType, Prover::num_vars>;
        /// Tag for dispatching to the right methods in cases where that depends on the non-dynamic phase method.
        using NonDynamicPhaseMethodTag = std::integral_constant<NonDynamicPhaseMethod,
                                                                ProverSettings::non_dynamic_phase_method>;
        /// Tag for CPU-based non-dynamic phase.
        using CPUTag = std::integral_constant<NonDynamicPhaseMethod, NonDynamicPhaseMethod::CPU>;

        /// Type of prover threads.
        using ThreadType = ProverThread<ProofRunner>;

        /// The context we use to evaluate expressions/predicates.
        struct Context {
            using NumberType = typename ProofRunner::NumberType;
            static constexpr bool analyze_monotonicity = false;
            static constexpr unsigned monotonicity_derivative_levels = 0;
            static constexpr unsigned irrational_precision = ProverSettings::irrational_precision;
        };

        /// The constructor.
        ProofRunner(const Prover* prover, const OnCritical* report_criticals, ProgressObserver* progress) :
            prover(prover),
            report_criticals(report_criticals),
            reported_criticals(0),
            cuboids_nothread(0),
            threads_done_with_init(0),
            num_threads(ProverSettings::cpu_threads()),
            queue(Variables::initial_dynamic_cuboids, num_threads),
            constraint_prop(&prover->vars, &prover->constraints),
            prover_threads{nullptr},
            progress(progress),
            progress_exit(false)
        {}

        /// The definition needs to be out-of-line; at this point, the ThreadType is still incomplete.
        inline ~ProofRunner();

        /// This method is mainly for testing purposes;
        /// it returns the entire cuboid queue content
        /// after initialization.
        std::vector<DynamicEntryType> generate_dynamic_phase() {
            run(true);
            return std::vector<DynamicEntryType>{queue.cbegin(), queue.cend()};
        }

        /// Actually run the proof (or only the initialization phase). Return true iff there were no criticals.
        bool run(bool init_only=false);

        std::size_t num_criticals() const noexcept {
            return reported_criticals;
        }

        std::size_t num_cuboids() const noexcept {
            return num_cuboids(NonDynamicPhaseMethodTag{});
        }

    private:
        /// Compute the number of cuboids if the proof was done on the CPU.
        inline std::size_t num_cuboids(CPUTag) const;

        /// Create an array of prover threads; must be done out-of-line due to imcomplete type.
        inline ThreadType* create_prover_threads();
        inline void destroy_prover_threads();

        /// IndexPack of non-dynamically split variables
        using StaticIndices = IndexRange<Variables::num_dynamic_vars, Variables::num_vars>;

        /// The prover thread needs access to our types/data.
        template<typename Runner> friend class ProverThread;

        /// Whether we are using the CPU to run the main phase of the proof.
        static constexpr bool use_cpu_threads = (ProverSettings::non_dynamic_phase_method == NonDynamicPhaseMethod::CPU);

        /// Number of subintervals in the first variable.
        static constexpr std::size_t first_var_subintervals = Variables::template NumSubdivisionsOf<0>::value;

        /// A type containing a compile-time constant variable index.
        template<std::size_t V> using VarIndex = std::integral_constant<std::size_t,V>;

        /// Join the prover threads. If the non-dynamic phase method is CPU, this waits for the proof to complete.
        /// Otherwise, this waits for the initialization to complete.
        void join_threads();

        /// Compute the number of subdivisions per dynamic split (i.e., num_dynamic_vars ** num_subdivisions_per_var).
        static constexpr std::size_t num_subs_per_dynamic_split(std::size_t num_vars, std::size_t num_subs_per_var) {
            return num_vars <= 1 ? num_subs_per_var :
                                   num_subs_per_dynamic_split(num_vars-1, num_subs_per_var) * num_subs_per_var;
        }

        /// Fill the array out of dynamic entries with the entries resulting from a dynamic split of split_entry.
        /// Begin filling at offset offset; increment offset while filling the array so that the final count of
        /// non-empty entries is stored in offset on return.
        template<std::size_t I1, std::size_t... Inds> inline
        void split_dynamic_fill_entries(const DynamicEntryType& split_entry, DynamicEntryType* out,
                                        std::size_t& offset, IndexPack<I1,Inds...>);
        inline void split_dynamic_fill_entries(const DynamicEntryType& split_entry, DynamicEntryType* out,
                                               std::size_t& offset, IndexPack<>);

        /// Split the dynamic variables of the given entry and push the resulting entries on the queue.
        inline void split_dynamic_vars(DynamicEntryType entry);

        /// Restrict the bounds of non-dynamically split variables in
        /// entry to the critical ranges in criticals_in_entry.
        static inline void restrict_non_dynamic_bounds(DynamicEntryType& entry,
                                                       const DynamicEntryType* criticals_in_entry);

        /// Decide whether we should split the dynamic variables of the given entry or if we can repeat work
        /// on the entry because the range of the non-dynamic variables was sufficiently reduced.
        static bool iterate_on_entry(const DynamicEntryType& entry, const DynamicEntryType* criticals_in_entry,
                                     std::size_t at_this_depth);

        template<typename V = void>
            void run_main_phase(std::integral_constant<NonDynamicPhaseMethod,NonDynamicPhaseMethod::CUDA>)
        {
            static_assert(!std::is_same<void,V>::value, "CUDA support is not implemented yet!");
        }

        void stop_progress_thread() {
            if(progress_thread.joinable()) {
                {
                    std::unique_lock<std::mutex> guard(progress_exit_mutex);
                    progress_exit = true;
                    progress_exit_cond.notify_one();
                }
                progress_thread.join();
            }
        }

        /// For method NonDynamicPhaseMethod::CPU, the main proof is also run by the initialization threads.
        void run_main_phase(std::integral_constant<NonDynamicPhaseMethod,NonDynamicPhaseMethod::CPU>) {}

        void progress_thread_observe() {
            std::unique_lock<std::mutex> guard(progress_exit_mutex);

            // do the initial wait
            std::chrono::milliseconds wait_initial{ProverSettings::initial_progress_wait_msec};
            std::chrono::milliseconds wait_interval{ProverSettings::progress_interval_msec};

            auto&& exit_condition = [&] () noexcept -> bool { return progress_exit; };

            progress_exit_cond.wait_for(guard, wait_initial, exit_condition);
            if(progress_exit) {
                return;
            }

            // do not call progress during initialization phase
            while(threads_done_with_init < num_threads) {
                progress_exit_cond.wait_for(guard, wait_interval, exit_condition);
                if(progress_exit) {
                    return;
                }
            }

            do {
                progress_thread_call_observer();
                progress_exit_cond.wait_for(guard, wait_interval, exit_condition);
            } while(!progress_exit);
        }

        void progress_thread_main() {
			SetRoundDown round_down;
			progress_thread_observe();
            progress->observe_done();
        }

        void progress_thread_call_observer() {
            const Context ctx{};
            DynamicEntryType entry;
            ProgressInfo info;
            info.critical_count = reported_criticals;
            info.cuboid_count = num_cuboids();
            info.queue_size = queue.size();
            if(queue.peek(entry)) {
                progress->observe_progress(ctx, entry, info);
            }
        }

        const Prover* prover;
        const OnCritical* report_criticals;
        std::atomic<std::size_t> reported_criticals;
        std::atomic<std::size_t> cuboids_nothread;
        std::atomic<std::size_t> threads_done_with_init;
        const std::size_t num_threads;
        CuboidQueue<DynamicEntryType, true, ProverSettings::cuboid_queue_order> queue;
        ConstraintPropagation<Variables,Constraints> constraint_prop;
        ThreadType* prover_threads;

        /// Progress reporting related.
        ProgressObserver* progress;
        std::thread progress_thread;
        std::mutex progress_exit_mutex;
        std::condition_variable progress_exit_cond;
        bool progress_exit;
    };
}

    template<typename Vars, typename Constrs>
    template<typename ProverSettings, typename ProgressObserver, typename OnCritical>
        bool Prover<Vars,Constrs>::run(const OnCritical& report_criticals, ProofInformation* information, ProgressObserver progress) const
    {
        impl::ProofRunner<Prover<Vars,Constrs>, ProverSettings, OnCritical, ProgressObserver>
            runner(this, &report_criticals, &progress);
        bool r = runner.run();
        if(information) {
            information->num_critical_cuboids = runner.num_criticals();
            information->num_cuboids = runner.num_cuboids();
        }
        return r;
    }

    template<typename Vars, typename Constrs> template<typename ProverSettings> auto Prover<Vars,Constrs>::
        generate_dynamic_phase_queue() const
            -> std::vector<impl::DynamicEntry<typename ProverSettings::NumberType, num_vars>>
    {
        const auto no_op = [] (const auto&, const auto&) -> void {};
		DefaultProgressObserver p;
        impl::ProofRunner<Prover<Vars,Constrs>, ProverSettings, std::decay_t<decltype(no_op)>, DefaultProgressObserver> runner(this, &no_op, &p);
        return runner.generate_dynamic_phase();
    }
}
