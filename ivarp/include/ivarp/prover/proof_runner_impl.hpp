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
//

#pragma once

template<typename P,typename S, typename O,typename G> ivarp::impl::ProofRunner<P,S,O,G>::~ProofRunner() {
    destroy_prover_threads();
}

template<typename P,typename S, typename O,typename G> void ivarp::impl::ProofRunner<P,S,O,G>::join_threads() {
    for(std::size_t i = 0; i < num_threads; ++i) {
        prover_threads[i].join();
    }
}

template<typename P,typename S, typename O,typename G>
std::size_t ivarp::impl::ProofRunner<P,S,O,G>::num_cuboids(CPUTag) const
{
    std::size_t c = cuboids_nothread;
    for(std::size_t i = 0; i < num_threads; ++i) {
        c += prover_threads[i].cuboids();
    }
    return c;
}

template<typename P,typename S, typename O,typename G> auto ivarp::impl::ProofRunner<P,S,O,G>::
        create_prover_threads() -> ThreadType*
{
    using Storage = std::aligned_storage_t<sizeof(ThreadType), alignof(ThreadType)>;
    auto* storage = new Storage[num_threads];
    auto* first = new (static_cast<void*>(&storage[0])) ThreadType(this, 0);
    for(std::size_t i = 1; i < num_threads; ++i) {
        new (static_cast<void*>(first+i)) ThreadType(this, i);
    }
    return first;
}

template<typename P,typename S, typename O,typename G> void ivarp::impl::ProofRunner<P,S,O,G>::
    destroy_prover_threads()
{
    using Storage = std::aligned_storage_t<sizeof(ThreadType), alignof(ThreadType)>;
    if(prover_threads) {
        for(std::size_t i = 0; i < num_threads; ++i) {
            prover_threads[i].~ThreadType();
        }
        auto* storage = reinterpret_cast<Storage*>(prover_threads);
        delete[] storage;
        prover_threads = nullptr;
    }
}

template<typename P, typename S, typename O,typename G> bool ivarp::impl::ProofRunner<P,S,O,G>::run(bool init_only) {
    // Create and start threads to initialize the cuboid queue; these threads also continue with the
    // main part of the proof if that is done on the CPU.
    destroy_prover_threads();

    {
        std::unique_lock<std::mutex> guard(progress_exit_mutex);
        progress_exit = false;
        progress_thread = std::thread(&ProofRunner::progress_thread_main, this);
    }

    prover_threads = create_prover_threads();
    for(std::size_t i = 0; i < num_threads; ++i) {
        prover_threads[i].start(init_only);
    }

    // either wait for just the initialization or initialization & proof to complete.
    join_threads();

    // the main proof may not be done yet
    if(!init_only) {
        run_main_phase(NonDynamicPhaseMethodTag{});
    }

    // stop the progress reporting thread
    stop_progress_thread();

    // return true iff there were no critical cuboids reported
    return reported_criticals == 0;
}
