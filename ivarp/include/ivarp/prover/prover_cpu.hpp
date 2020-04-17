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
// Created by Phillip Keldenich on 19.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename P,typename S,typename O,typename G>
        void ProofRunner<P,S,O,G>::split_dynamic_vars(DynamicEntryType entry)
    {
        constexpr std::size_t num_subs =
            num_subs_per_dynamic_split(Variables::num_dynamic_vars,
                                       ProverSettings::subintervals_per_dynamic_var_split);
        DynamicEntryType entries[num_subs];
        std::size_t offset = 0;
        entry.depth += 1;
        split_dynamic_fill_entries(entry, entries, offset, IndexRange<0,Variables::num_dynamic_vars>{});

        queue.enqueue_bulk(+entries, entries+offset);
    }

    template<typename P,typename S,typename O,typename G>
    template<std::size_t I1, std::size_t... Inds> void ProofRunner<P,S,O,G>::
        split_dynamic_fill_entries(const DynamicEntryType& split_entry, DynamicEntryType* out,
                                   std::size_t& offset, IndexPack<I1,Inds...>)
    {
        constexpr std::size_t apps = ProverSettings::dynamic_phase_propagation_applications_per_constraint;
        Splitter<NumberType> subs(split_entry.bounds[I1], ProverSettings::subintervals_per_dynamic_var_split);
        for(int i = 0; i < ProverSettings::subintervals_per_dynamic_var_split; ++i) {
            DynamicEntryType entry = split_entry;
            entry.bounds[I1] = subs.subrange(i);
            if(constraint_prop.template restrict_var_bounds<I1,Variables::num_vars,Context>(entry.bounds).empty) {
                ++cuboids_nothread;
                continue;
            }
            if(constraint_prop.template dynamic_post_split_var<I1,apps,Context,
                                                               Variables::num_vars>(entry.bounds).empty)
            {
                ++cuboids_nothread;
                continue;
            }
            split_dynamic_fill_entries(entry, out, offset, IndexPack<Inds...>{});
        }
    }

    template<typename P,typename S,typename O,typename G> void ProofRunner<P,S,O,G>::
        split_dynamic_fill_entries(const DynamicEntryType& split_entry, DynamicEntryType* out,
                                   std::size_t& offset, IndexPack<>)
    {
        out[offset++] = split_entry;
    }

    /// Restrict the bounds of non-dynamically split variables in
    /// entry to the critical ranges in criticals_in_entry.
    template<typename P,typename S,typename O,typename G> void ProofRunner<P,S,O,G>::
        restrict_non_dynamic_bounds(DynamicEntryType& entry, const DynamicEntryType* criticals_in_entry)
    {
        for(std::size_t i = Variables::num_dynamic_vars; i < Variables::num_vars; ++i) {
            entry.bounds[i] = criticals_in_entry->bounds[i];
        }
    }

    template<typename P,typename S,typename O,typename G> bool ProofRunner<P,S,O,G>::
        iterate_on_entry(const DynamicEntryType& entry, const DynamicEntryType* criticals_in_entry,
                         std::size_t at_this_depth)
    {
        if(entry.depth == 0 && at_this_depth == 0) {
            return true;
        }
        if(at_this_depth >= ProverSettings::max_iterations_per_depth) {
            return false;
        }

        // otherwise, iterate if at least one variable range was cut in half
        for(std::size_t i = Variables::num_dynamic_vars; i < Variables::num_vars; ++i) {
            auto old_width = entry.bounds[i].ub() - entry.bounds[i].lb();
            auto new_width = criticals_in_entry->bounds[i].ub() - criticals_in_entry->bounds[i].lb();
            if(2*new_width <= old_width) {
                return true;
            }
        }
        return false;
    }
}
}
