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
// Created by Phillip Keldenich on 21.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename ProverInputType, typename OnCritical> class CPUProverCore :
        public BasicProverCore<ProverInputType, OnCritical>
    {
    public:
        using Base = BasicProverCore<ProverInputType, OnCritical>;
        using Context = typename ProverInputType::Context;
        using NumberType = typename Context::NumberType;
        using ConcreteNT = typename NumberType::NumberType;
        using RBT = typename ProverInputType::RuntimeBoundTable;
        using RCT = typename ProverInputType::RuntimeConstraintTable;
        using DBA = DynamicBoundApplication<RBT, Context>;
        using StaticSplitInfo = typename ProverInputType::StaticSplitInfo;
        using QueueElement = ProofDriverQueueEntry<NumberType, ProverInputType::num_args>;

        static constexpr std::size_t num_args = ProverInputType::num_args;

        explicit CPUProverCore() = default;

        void initialize(std::size_t /* num_threads */) const noexcept {}
        void initialize_per_thread(std::size_t /*cpu_thread_id*/, std::size_t /*num_threads*/) const noexcept {}

        CuboidCounts handle_cuboids_nonfinal(std::size_t /*thread_id*/, const std::vector<QueueElement>& input_cuboids,
                                             std::vector<QueueElement>* output_cuboids) const
        {
            CuboidCounts counts;
            for(const QueueElement& e : input_cuboids) {
                p_handle_cuboid_nf(e, output_cuboids, &counts);
            }
            return counts;
        }

        CuboidCounts handle_cuboids_final(std::size_t /*thread_id*/,
                                          const std::vector<QueueElement>& final_elements) const
        {
            CuboidCounts counts;
            CallOnCritical coc{this->on_critical};
            for(const QueueElement& e : final_elements) {
                this->p_handle_cuboid(coc, e, &counts, StaticSplitInfo{});
            }
            return counts;
        }

        void set_settings(const ProverSettings* settings) override {
            this->Base::set_settings(settings);
            iteration_volume = convert_number<ConcreteNT>(settings->iteration_volume_criterion);
            iteration_single = convert_number<ConcreteNT>(settings->iteration_single_dimension_criterion);
        }

        void replace_default_settings(ProverSettings& settings) const {
            if(settings.dequeue_buffer_size <= 0) {
                settings.dequeue_buffer_size = 32;
            }
            if(settings.max_iterations_per_node < 0) {
                settings.max_iterations_per_node = 4;
            }
            if(settings.iteration_single_dimension_criterion < 0) {
                settings.iteration_single_dimension_criterion = 0.66f;
            }
            if(settings.iteration_volume_criterion < 0) {
                settings.iteration_volume_criterion = 0.25f;
            }
        }

    private:
        /// Action used for non-final elements; the elements in the same dynamic element are collected and
        /// their ranges for each argument are joined.
        struct AddToRange {
            NumberType intervals[num_args];
            bool empty{true};

            void handle_critical(const QueueElement& e, CuboidCounts*) {
                if(empty) {
                    empty = false;
                    std::copy_n(+e.elements, num_args, +intervals);
                } else {
                    for(std::size_t i = 0; i < num_args; ++i) {
                        intervals[i].do_join_defined(e.elements[i]);
                    }
                }
            }
        };

        void p_handle_cuboid_nf(QueueElement e, std::vector<QueueElement>* output_cuboids, CuboidCounts* counts) const {
            int it = 0;
            for(;;) {
                AddToRange add_to_range;
                this->p_handle_cuboid(add_to_range, e, counts, StaticSplitInfo{});
                if(add_to_range.empty) {
                    return;
                }

                if(++it >= this->settings->max_iterations_per_node || !p_should_iterate(add_to_range, e)) {
                    output_cuboids->emplace_back();
                    QueueElement &out_e = output_cuboids->back();
                    std::copy_n(+add_to_range.intervals, num_args, +out_e.elements);
                    out_e.depth = e.depth;
                    return;
                }

                counts->num_repeated_nodes++;
                counts->num_cuboids++;
                std::copy_n(+add_to_range.intervals, num_args, +e.elements);
            }
        }

        /// Action used for final elements; calls on_critical.
        struct CallOnCritical {
            explicit CallOnCritical(const OnCritical* on_critical) :
                on_critical(on_critical)
            {}

            void handle_critical(const QueueElement& e, CuboidCounts* counts) const {
                counts->num_critical_cuboids += 1;
                (*on_critical)(Context{}, +e.elements);
            }

        private:
            const OnCritical* on_critical;
        };

        bool p_should_iterate(const AddToRange& new_ranges, const QueueElement& old_ranges) const {
            return p_should_iterate_single_dim(new_ranges, old_ranges) ||
                   p_should_iterate_volume(new_ranges, old_ranges);
        }

        bool p_should_iterate_single_dim(const AddToRange& new_ranges, const QueueElement& old_ranges) const {
            for(std::size_t i = 0; i < num_args; ++i) {
                ConcreteNT mnw = new_ranges.intervals[i].lb() - new_ranges.intervals[i].ub();
                ConcreteNT mow = old_ranges.elements[i].lb() - old_ranges.elements[i].ub();
                if(mow * iteration_single > mnw) {
                    return true;
                }
            }
            return false;
        }

        bool p_should_iterate_volume(const AddToRange& new_ranges, const QueueElement& old_ranges) const {
            ConcreteNT mnv = new_ranges.intervals[0].lb() - new_ranges.intervals[0].ub();
            ConcreteNT mov = old_ranges.elements[0].lb() - old_ranges.elements[0].ub();
            for(std::size_t i = 1; i < num_args; ++i) {
                ConcreteNT ncw = new_ranges.intervals[i].ub() - new_ranges.intervals[i].lb();
                ConcreteNT ocw = old_ranges.elements[i].ub() - old_ranges.elements[i].lb();
                mnv *= ncw;
                mov *= ocw;
            }
            return mov * iteration_volume > mnv;
        }

        template<std::size_t Arg> bool p_cuboid_violates_constraint_row(const QueueElement& e) const {
            bool all_sat = true;
            const auto& constraints = ivarp::template get<Arg>(*this->rct);
            auto visitor = [&](const auto& constr) -> TupleVisitationControl {
                if (!possibly(constr.template array_evaluate<Context>(+e.elements))) {
                    all_sat = false;
                    return TupleVisitationControl::STOP;
                }
                return TupleVisitationControl::CONTINUE;
            };
            visit_tuple(visitor, constraints);
            return !all_sat;
        }

        template<std::size_t CurrArg, typename NextSplit, typename... Splits,
                 std::enable_if_t<CurrArg==NextSplit::arg,int> = 0>
            bool p_cuboid_violates_constraints(const QueueElement& e, SplitInfoSequence<NextSplit,Splits...>) const
        {
            return false;
        }

        template<std::size_t CurrArg, std::enable_if_t<CurrArg==num_args,int> = 0>
            bool p_cuboid_violates_constraints(const QueueElement& e, SplitInfoSequence<>) const
        {
            return false;
        }

        template<std::size_t CurrArg, typename NextSplit, typename... Splits,
                 std::enable_if_t<(CurrArg < NextSplit::arg),int> = 0>
            bool p_cuboid_violates_constraints(const QueueElement& e, SplitInfoSequence<NextSplit,Splits...> s) const
        {
            if(p_cuboid_violates_constraint_row<CurrArg>(e)) { return true; }
            return p_cuboid_violates_constraints<CurrArg+1>(e,s);
        }

        template<std::size_t CurrArg, std::enable_if_t<(CurrArg < num_args),int> = 0>
            bool p_cuboid_violates_constraints(const QueueElement& e, SplitInfoSequence<> s) const
        {
            if(p_cuboid_violates_constraint_row<CurrArg>(e)) { return true; }
            return p_cuboid_violates_constraints<CurrArg+1>(e,s);
        }

        template<typename ActionOnCritical, typename S1, typename... Splits>
            void p_handle_cuboid(ActionOnCritical& action, const QueueElement& e,
                                 CuboidCounts* counts, SplitInfoSequence<S1,Splits...>) const
        {
            constexpr int subs = static_cast<int>(S1::subdivisions);
            constexpr std::size_t arg = S1::arg;
            using BoundEv = BoundEvent<arg, BoundID::BOTH>;

            Splitter<NumberType> splitter(e.elements[arg], subs);
            for(NumberType n : splitter) {
                QueueElement qe{e};
                qe.elements[arg] = n;
                if(propagate_iterated_recursive::propagate<BoundEv, Context>(*this->rbt, qe.elements, *this->dba,
                                                                             num_args, 4).empty)
                {
                    counts->num_leaf_cuboids++;
                    continue;
                }

                if(p_cuboid_violates_constraints<arg>(qe, SplitInfoSequence<Splits...>{})) {
                    counts->num_leaf_cuboids++;
                    continue;
                }

                p_handle_cuboid(action, qe, counts, SplitInfoSequence<Splits...>{});
            }
            counts->num_cuboids += subs;
        }

        template<typename ActionOnCritical>
            void p_handle_cuboid(ActionOnCritical& action, const QueueElement& e,
                                 CuboidCounts* counts, SplitInfoSequence<>) const
        {
            counts->num_leaf_cuboids++;
            action.handle_critical(e, counts);
        }

        ConcreteNT iteration_volume;
        ConcreteNT iteration_single;
    };
}
}
