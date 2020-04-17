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

#pragma once

#include <chrono>
#include <thread>
#include "ivarp/prover/cuboid_queue.hpp"
#include "ivarp/bound_propagation.hpp"
#include "ivarp/splitter.hpp"
#include <boost/range/iterator_range.hpp>

namespace ivarp {
namespace impl {
    template<typename NumberType, std::size_t NumArgs> struct ProofDriverQueueEntry {
        IVARP_DEFAULT_CM(ProofDriverQueueEntry);

        ProofDriverQueueEntry() = default;
        explicit IVARP_HD ProofDriverQueueEntry(const Array<NumberType,NumArgs>& a) : depth(0) {
            for(std::size_t i = 0; i < NumArgs; ++i) {
                elements[i] = a[i];
            }
        }

        NumberType elements[NumArgs];
        std::size_t depth;
    };

    struct CuboidCounts {
        std::size_t num_cuboids{0};
        std::size_t num_leaf_cuboids{0};
        std::size_t num_critical_cuboids{0};
        std::size_t num_repeated_nodes{0};

        IVARP_HD CuboidCounts &operator+=(const CuboidCounts& c) noexcept {
            num_cuboids += c.num_cuboids;
            num_leaf_cuboids += c.num_leaf_cuboids;
            num_critical_cuboids += c.num_critical_cuboids;
            num_repeated_nodes += c.num_repeated_nodes;
            return *this;
        }

#if defined(__CUDA_ARCH__)
        IVARP_D void gpu_atomic_add(const CuboidCounts& c) noexcept {
			ivarp::gpu_atomic_add(&num_cuboids, c.num_cuboids);
			ivarp::gpu_atomic_add(&num_leaf_cuboids, c.num_leaf_cuboids);
			ivarp::gpu_atomic_add(&num_critical_cuboids, c.num_critical_cuboids);
			ivarp::gpu_atomic_add(&num_repeated_nodes, c.num_repeated_nodes);
        }
#endif
    };

    static inline IVARP_HD ProofInformation& operator+=(ProofInformation& p, const CuboidCounts& c) noexcept {
        p.num_cuboids += c.num_cuboids;
        p.num_leaf_cuboids += c.num_leaf_cuboids;
        p.num_critical_cuboids += c.num_critical_cuboids;
        p.num_repeated_nodes += c.num_repeated_nodes;
        return p;
    }

    /**
     * An object that captures the state of the proof in a proof driver.
     */
    class ProofDriverState {
    public:
        /**
         * Create a new ProofDriverState in its initial state.
         *
         * @param num_threads
         * @param queue_mutex
         */
        ProofDriverState(std::size_t num_threads, std::mutex* queue_mutex) noexcept :
            mutex(queue_mutex), num_threads(num_threads)
        {}

        /**
         * Re-throws the error, if there was any, on the thread calling this method.
         * Does not have to take a lock.
         */
        void reraise_error_if_any() const {
            if(have_error.load() == 2) {
                std::rethrow_exception(error);
            }
        }

        /**
         * Set an error. The error is only ever set once.
         * @param error The error to set.
         * @param error_thread The thread ID raising the error.
         */
        void set_error(std::exception_ptr error, std::size_t error_thread) noexcept {
            int expected = 0;
            if(have_error.compare_exchange_strong(expected, 1)) {
                this->error = error;
                this->error_thread = error_thread;
                have_error.store(2);
                state_changed.notify_all();
            }
        }

        /**
         * Check whether there is an error. Does not need to lock.
         * @return
         */
        bool error_is_set() const noexcept {
            return have_error.load() == 2;
        }

        std::size_t get_error_thread() const noexcept {
            return error_thread;
        }

        /**
         * Declare that the thread calling this method is done with proof initialization.
         */
        void init_done() noexcept {
            std::unique_lock<std::mutex> lock(*mutex);
            if(++threads_done_init >= num_threads) {
                state_changed.notify_all();
            }
        }

        /**
         * Wait for all threads to be done with proof initialization.
         * Reraises the error if any is set.
         */
        void wait_for_init() const {
            std::unique_lock<std::mutex> lock(*mutex);
            state_changed.wait(lock, [&]() -> bool {
                return error_is_set() || threads_done_init >= num_threads;
            });
            reraise_error_if_any();
        }

        /**
         * Declare that queue compaction is done.
         */
        void queue_compaction_done() noexcept {
            std::unique_lock<std::mutex> lock(*mutex);
            compaction_done = true;
            state_changed.notify_all();
        }

        /**
         * Wait for queue compaction to complete.
         * Re-raises the error if any is set.
         */
         void wait_for_queue_compaction() const {
            std::unique_lock<std::mutex> lock(*mutex);
            state_changed.wait(lock, [&]() -> bool {
                return error_is_set() || compaction_done;
            });
            reraise_error_if_any();
         }

         /**
          * Declare that the proof is done. Only called by the main thread.
          */
         void declare_proof_done() noexcept {
            std::unique_lock<std::mutex> lock(*mutex);
            proof_done = true;
            state_changed.notify_all();
         }

         /**
          * The method used by the progress observer to wait for:
          *   * a specific time limit to occur,
          *   * an error being set,
          *   * the proof being done.
          * Uses an external lock. Returns true if the proof was done in time.
          * Re-raises the error if any is set.
          */
         template<typename TimePoint>
            bool wait_until_done(std::unique_lock<std::mutex>& lock, TimePoint wait_until)
         {
            bool status = state_changed.wait_until(lock, wait_until, [&] () -> bool {
                return error_is_set() || proof_done;
            });

            if(!status) {
                return false;
            }
            reraise_error_if_any();
            return true;
         }

    private:
        std::mutex* mutex; ///< The mutex to use for waiting and other situations where locking is necessary.
        mutable std::condition_variable state_changed; ///< A condition variable to wait for state changes.
        std::size_t num_threads; ///< The number of threads running the proof.
        std::size_t threads_done_init{0}; ///< The number of threads done with proof initialization.
        std::atomic<int> have_error{0}; ///< Whether we have encountered an exception on any thread. 0: no error, 1: error is being set, 2: error is set.
        std::exception_ptr error; ///< The error, if any.
        std::size_t error_thread; ///< The thread that set the error.
        bool compaction_done{false}; ///< Whether queue compaction after initialization is done.
        bool proof_done{false}; ///< Whether the proof is done.
    };

    template<typename ProverInputType, typename CoreProver, typename OnCritical, typename ProgressReporter>
    class ProofDriver
    {
    private:
        // Number of variables and args.
        static constexpr std::size_t num_args = ProverInputType::num_args;
        static constexpr std::size_t num_vars = ProverInputType::num_vars;
        static constexpr std::size_t initial_queue_size = ProverInputType::initial_queue_size;

        // Prover input types
        using Context = typename ProverInputType::Context;
        using DynamicSplitInfo = typename ProverInputType::DynamicSplitInfo;
        using StaticSplitInfo = typename ProverInputType::StaticSplitInfo;
        using RBT = typename ProverInputType::RuntimeBoundTable;
        using RCT = typename ProverInputType::RuntimeConstraintTable;

        // Propagation information
        using DBA = DynamicBoundApplication<RBT, Context>;

        // Number-type and queue related types
        using NumberType = typename Context::NumberType;
        using QueueElement = ProofDriverQueueEntry<NumberType, num_args>;
        using QueueType = CuboidQueue<QueueElement, true, CuboidQueueOrder::LIFO>;
        using Lock = std::unique_lock<std::mutex>;

        struct ProofDriverThread {
            ProofDriverThread(ProofDriver* driver, std::size_t id) :
                driver(driver),
                id(id),
                handle()
            {
                dequeue_buffer.reserve(driver->settings.dequeue_buffer_size);
            }

            ProofDriverThread(const ProofDriverThread&) = delete;
            ProofDriverThread &operator=(const ProofDriverThread&) = delete;

            ProofDriverThread(ProofDriverThread&& o) noexcept :
                driver(o.driver),
                id(o.id),
                handle(std::move(o.handle)),
                dequeue_buffer(std::move(o.dequeue_buffer)),
                result_buffer(std::move(o.result_buffer))
            {}

            ProofDriverThread &operator=(ProofDriverThread&& o) noexcept {
                driver = o.driver;
                id = o.id;
                handle = std::move(o.handle);
                dequeue_buffer.swap(o.dequeue_buffer);
                result_buffer.swap(o.result_buffer);
                return *this;
            }

            void join() {
                if(id != 0 && handle.joinable()) {
                    handle.join();
                }
            }

            ~ProofDriverThread() {
                if(handle.joinable()) {
                    handle.join();
                }
            }

            void entry_point() noexcept {
                try {
                    proof_init();
                    driver->state.init_done();
                    driver->state.wait_for_queue_compaction();
                    proof_main();
                } catch(...) {
                    std::exception_ptr cexp = std::current_exception();
                    driver->state.set_error(cexp, id);
                }
            }

            void spawn() {
                assert(!handle.joinable());
                handle = std::thread(&ProofDriverThread::entry_point, this);
                assert(handle.joinable());
            }

            void proof_init() {
                proof_init_begin(driver->initial_cuboid, DynamicSplitInfo{});
            }

            static constexpr auto rec_limit = static_cast<std::uint8_t>(ivarp::min(num_args,255));
            static constexpr auto it_limit = 2*num_args;

            void proof_init_set_to_empty(std::size_t count, std::size_t* offset) const noexcept {
                // mark as empty by making the first interval empty; all empty entries will be compacted
                // after the initial phase to allow parallel initialization
                for(std::size_t j = 0; j < count; ++j) {
                    QueueElement& qe = driver->queue[*offset];
                    qe.elements[0].set_lb(1);
                    qe.elements[0].set_ub(0);
                    *offset += 1;
                }
            }

            template<typename S1, typename... Splits>
                void proof_init_begin(const Array<NumberType,num_args>& initial, const SplitInfoSequence<S1,Splits...>&)
            {
                using BoundEv = BoundEvent<S1::arg, BoundID::BOTH>;
                const auto& rbt = driver->rbt;
                const auto& dba = driver->dba;
                const auto num_threads = static_cast<std::size_t>(driver->settings.thread_count);
                constexpr std::size_t s1s = S1::initial;
                constexpr std::size_t rem_initial = initial_queue_size / s1s;

                const std::size_t s1s_per_thread = ivarp::max(s1s / num_threads, 1);
                if(id >= s1s) {
                    return;
                }

                const std::size_t beg = id * s1s_per_thread;
                const std::size_t end = (id != num_threads-1) ? (id+1) * s1s_per_thread : s1s;
                Splitter<NumberType> splitter(initial[S1::arg], static_cast<int>(s1s));
                std::size_t offset = beg * rem_initial;
                for(int i = static_cast<int>(beg); i < static_cast<int>(end); ++i) {
                    ++counts.num_cuboids;
                    QueueElement qsub{initial};
                    qsub.elements[S1::arg] = splitter.subrange(i);
                    if(propagate_iterated_recursive::propagate<BoundEv,Context>(rbt, qsub.elements, dba,
                                                                                it_limit, rec_limit).empty)
                    {
                        ++counts.num_leaf_cuboids;
                        proof_init_set_to_empty(rem_initial, &offset);
                    } else {
                        proof_init_continue<rem_initial>(qsub, &offset, SplitInfoSequence<Splits...>{});
                    }
                }
            }

            template<std::size_t EntriesPerElement, typename SN, typename... Splits>
            void proof_init_continue(const QueueElement& e, std::size_t* offset, const SplitInfoSequence<SN,Splits...>&)
            {
                using BoundEv = BoundEvent<SN::arg, BoundID::BOTH>;
                const auto& rbt = driver->rbt;
                const auto& dba = driver->dba;
                constexpr std::size_t sns = SN::initial;
                constexpr std::size_t next_epe = EntriesPerElement / sns;
                Splitter<NumberType> splitter(e.elements[SN::arg], static_cast<int>(sns));
                for(NumberType sub : splitter) {
                    ++counts.num_cuboids;
                    QueueElement qsub{e};
                    qsub.elements[SN::arg] = sub;
                    if(propagate_iterated_recursive::propagate<BoundEv, Context>(rbt, qsub.elements, dba,
                                                                                 it_limit, rec_limit).empty)
                    {
                        ++counts.num_leaf_cuboids;
                        proof_init_set_to_empty(next_epe, offset);
                    } else {
                        proof_init_continue<next_epe>(qsub, offset, SplitInfoSequence<Splits...>{});
                    }
                }
            }

            template<std::size_t EntriesPerElement>
            void proof_init_continue(const QueueElement& e, std::size_t* offset, const SplitInfoSequence<>&) {
                static_assert(EntriesPerElement == 1, "Something is wrong with the initial queue size!");
                QueueElement& qe = driver->queue[*offset];
                *offset += 1;

                if(!driver->satisfies_all_constraints(e)) {
                    qe.elements[0].set_lb(1);
                    qe.elements[0].set_ub(0);
                    ++counts.num_leaf_cuboids;
                } else {
                    qe = e;
                }
            }

            void proof_main() {
                driver->core->initialize_per_thread(id, static_cast<std::size_t>(driver->settings.thread_count));
                while(refill_buffer()) {
                    driver->state.reraise_error_if_any();
                    counts += driver->core->handle_cuboids_nonfinal(id, dequeue_buffer, &result_buffer);
                    dequeue_buffer.clear();
                    if(!result_buffer.empty()) {
                        handle_core_result();
                    }
                }
            }

            bool refill_buffer() {
                driver->queue.pop_into(dequeue_buffer, driver->settings.dequeue_buffer_size);
                return !dequeue_buffer.empty();
            }

            void handle_core_result() {
                std::vector<QueueElement> final_elements;
                for(const QueueElement& q : result_buffer) {
                    if(q.depth >= driver->settings.generation_count) {
                        final_elements.push_back(q);
                    } else {
                        dequeue_buffer.push_back(q);
                    }
                }
                result_buffer.clear();

                if(!final_elements.empty()) {
                    counts += driver->core->handle_cuboids_final(id, final_elements);
                }
                if(!dequeue_buffer.empty()) {
                    split_and_requeue();
                }
            }

            void split_and_requeue() {
                for(const QueueElement& e : dequeue_buffer) {
                    split_into(e, result_buffer);
                }
                dequeue_buffer.clear();
                if(!result_buffer.empty()) {
                    driver->queue.enqueue_bulk(result_buffer.begin(), result_buffer.end());
                }
                result_buffer.clear();
            }

            struct SplitIntoImpl;
            void split_into(QueueElement e, std::vector<QueueElement>& output);

            ProofDriver* driver;
            std::size_t id;
            std::thread handle;
            std::vector<QueueElement> dequeue_buffer, result_buffer;
            CuboidCounts counts;
        };

    public:
        explicit ProofDriver(ProverInputType input, const OnCritical* on_c,
                             ProgressReporter* rep, ProofInformation* info,
                             CoreProver* core, ProverSettings s) noexcept :
            rbt(ivarp::move(input.runtime_bounds)),
            dba(&rbt),
            rct(ivarp::move(input.runtime_constraints)),
            on_critical(on_c), reporter(rep),
            information(info), core(core), settings(ivarp::move(s)),
            initial_cuboid(ivarp::move(input.initial_runtime_bounds)),
            queue(initial_queue_size, static_cast<unsigned>(settings.thread_count)),
            state(static_cast<std::size_t>(settings.thread_count), &queue.mutex())
        {
            core->set_runtime_bounds(&rbt);
            core->set_runtime_constraints(&rct);
            core->set_dynamic_bound_application(&dba);
            core->set_on_critical(on_critical);
            core->set_settings(&settings);
            core->initialize(static_cast<std::size_t>(settings.thread_count));
            p_init_threads();
        }

        bool run() {
            try {
                p_start_progress();
                p_spawn_threads();
                threads.front().proof_init();
                p_compact_initial_queue();
                threads.front().proof_main();
                p_join_threads();
                state.declare_proof_done();
            } catch(...) {
                state.set_error(std::current_exception(), 0);
                p_join_threads();
            }

            progress_observer.join();
            state.reraise_error_if_any();
            return information->num_critical_cuboids == 0;
        }

        bool satisfies_all_constraints(const QueueElement& e) const {
            return satisfies_all_constraints(e.elements);
        }

        template<typename ArrayType>
        bool satisfies_all_constraints(const ArrayType& values) const {
            const auto& all_c = ProverInputType::all_constraints(rct);
            bool all_sat = true;
            auto visitor = [&] (const auto& constr) -> TupleVisitationControl {
                auto cres = constr.template array_evaluate<Context>(values);
                if(!possibly(cres)) {
                    all_sat = false;
                    return TupleVisitationControl::STOP;
                }
                return TupleVisitationControl::CONTINUE;
            };
            visit_tuple(visitor, all_c);
            return all_sat;
        }

    private:
        void p_spawn_threads() {
            for(ProofDriverThread& t : boost::make_iterator_range(threads.begin() + 1, threads.end())) {
                t.spawn();
            }
        }

        void p_compact_initial_queue() {
            state.init_done();
            state.wait_for_init();
            {
                Lock lock(queue.mutex());
                auto b = queue.begin();
                auto e = queue.end();
                auto new_end = std::remove_if(b, e, [] (const auto& e) { return e.elements[0].empty(); });
                queue.unlocked_erase(new_end, e);
            }
            state.queue_compaction_done();
        }

        void p_join_threads() {
            information->num_critical_cuboids = 0;
            information->num_cuboids = 0;
            information->num_leaf_cuboids = 0;
            information->num_repeated_nodes = 0;
            for(ProofDriverThread& t : threads) {
                t.join();
                *information += t.counts;
            }
        }

        void p_init_threads() {
            const auto nt = static_cast<std::size_t>(settings.thread_count);
            threads.reserve(nt);
            for(std::size_t i = 0; i < nt; ++i) {
                threads.emplace_back(this, i);
            }
        }

        // Progress observation.
        void p_start_progress() {
            progress_observer = std::thread(&ProofDriver::p_progress_main, this);
        }

        void p_progress_main() noexcept {
            try {
                auto begun_at = std::chrono::steady_clock::now();
                auto begin_progress_at = begun_at + settings.start_progress_after;

                // wait for initialization & compaction
                state.wait_for_queue_compaction();

                // take lock
                Lock lock{queue.mutex()};
                if(state.wait_until_done(lock, begin_progress_at)) {
                    p_call_progress_observer_done();
                    return;
                }

                for(;;) {
                    p_call_progress_observer();
                    auto next_progress_at = std::chrono::steady_clock::now() + settings.progress_every;
                    if(state.wait_until_done(lock, next_progress_at)) {
                        p_call_progress_observer_done();
                        return;
                    }
                }
            } catch(...) {
                p_call_progress_observer_error(std::current_exception(), state.get_error_thread());
            }
        }

        void p_call_progress_observer() {
            ProgressInfo info;
            info.queue_size = queue.unlocked_size();
            for(const ProofDriverThread& t : threads) {
                info.leaf_count += t.counts.num_leaf_cuboids;
                info.critical_count += t.counts.num_critical_cuboids;
                info.cuboid_count += t.counts.num_cuboids;
            }
            QueueElement element;
            if(queue.unlocked_peek(element)) {
                reporter->observe_progress(Context{}, element, info);
            }
        }

        void p_call_progress_observer_done() {
            reporter->observe_done();
        }

        void p_call_progress_observer_error(std::exception_ptr error, std::size_t error_thread_id) {
            reporter->observe_error(error, error_thread_id);
        }

        // Progress observer thread.
        std::thread progress_observer;

        // Immutable information, such as settings, bounds/constraints.
        const RBT rbt;
        const DBA dba;
        const RCT rct;
        const OnCritical* const on_critical;
        ProgressReporter* const reporter;
        ProofInformation* const information;
        CoreProver* const core;
        ProverSettings const settings;
        Array<NumberType, num_args> const initial_cuboid;

        // The queue and per-thread information.
        QueueType queue;

        // The global state of the proof, i.e., what stage is running, etc.
        ProofDriverState state;

        // The threads running the proof, along with their state.
        std::vector<ProofDriverThread> threads;
    };
}
}

#include "pd_impl_split_into.hpp"
