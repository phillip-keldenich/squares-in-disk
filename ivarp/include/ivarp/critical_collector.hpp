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
// Created by Phillip Keldenich on 15.04.20.
//

#pragma once

#include "ivarp/array.hpp"
#include "ivarp/prover_input.hpp"
#include <vector>

namespace ivarp {
    template<typename ProverInputType> using CriticalCollection =
        std::vector<Array<typename ProverInputType::NumberType, ProverInputType::num_args>>;

    template<typename ProverInputType> class CriticalCollector {
    public:
        using Context = typename ProverInputType::Context;
        using Collection = CriticalCollection<ProverInputType>;

        explicit CriticalCollector(Collection* collection, std::mutex* lock = nullptr) :
            collection(collection),
            lock(lock)
        {}

        template<typename Crit> void operator()(const Context&, const Crit& c) const {
            if(lock) {
                std::unique_lock<std::mutex> l(*lock);
                p_do_add(c);
            } else {
                p_do_add(c);
            }
        }

    private:
        template<typename Crit> void p_do_add(const Crit& c) const {
            collection->emplace_back();
            collection->back().initialize_from(&c[0]);
        }

        Collection *collection;
        std::mutex *lock;
    };
}
