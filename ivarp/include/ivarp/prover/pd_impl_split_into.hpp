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
// Created by Phillip Keldenich on 14.02.20.
//

#pragma once

template<typename PIT, typename CP, typename OC, typename PR>
struct ivarp::impl::ProofDriver<PIT, CP, OC, PR>::ProofDriverThread::SplitIntoImpl
{
    using Splitter = ivarp::Splitter<NumberType>;

    static constexpr std::size_t itlim = 2 * num_args;
    static constexpr std::uint8_t reclim = std::uint8_t(ivarp::min(num_args, 32));

    template<typename S1, typename... Splits>
    static void perform_split(ProofDriverThread& t, const QueueElement& e,
                              std::vector<QueueElement>& output, SplitInfoSequence<S1, Splits...>)
    {
        using BoundEv = BoundEvent<S1::arg, BoundID::BOTH>;
        using propagate_iterated_recursive::propagate;
        const auto& rbt = t.driver->rbt;
        const auto& dba = t.driver->dba;
        Splitter splitter{e.elements[S1::arg], S1::subdivisions};
        for(NumberType sub : splitter) {
            ++t.counts.num_cuboids;
            QueueElement x{e};
            x.elements[S1::arg] = sub;
            if(propagate<BoundEv,Context>(rbt, x.elements, dba, itlim, reclim).empty) {
                ++t.counts.num_leaf_cuboids;
                continue;
            }
            perform_split(t, x, output, SplitInfoSequence<Splits...>{});
        }
    }

    static void perform_split(ProofDriverThread& t, const QueueElement& e,
                              std::vector<QueueElement>& output, SplitInfoSequence<>)
    {
        if(t.driver->satisfies_all_constraints(e)) {
            output.push_back(e);
        } else {
            ++t.counts.num_leaf_cuboids;
        }
    }
};

template<typename PIT, typename CP, typename OC, typename PR>
void ivarp::impl::ProofDriver<PIT, CP, OC, PR>::ProofDriverThread::split_into(QueueElement e,
                                                                              std::vector<QueueElement> &output)
{
    e.depth += 1;
    SplitIntoImpl::perform_split(*this, e, output, DynamicSplitInfo{});
}
