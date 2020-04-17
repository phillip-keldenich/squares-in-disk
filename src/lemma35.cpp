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
// Created by Phillip Keldenich on 03.12.19.
//

#include "auxiliary_functions.hpp"
#include "proof_auxiliaries.hpp"
#include "lemma35_proof.hpp"

template<unsigned N> static void run_lemma35_N() {
    std::ostringstream proof_name;
    proof_name << "Lemma 35, N = " << N;

    const auto system = lemma35_proof::System<N>::make_system();
    const auto input = prover_input<CTX, lemma35_proof::VarSplit>(system);
    const auto printer = ivarp::critical_printer(std::cerr, system);
    run_proof(proof_name.str(), input, system, printer);
}

void run_lemma35() {
    run_lemma35_N<5>();
    run_lemma35_N<6>();
    run_lemma35_N<7>();
}
