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
#include "lemma34_proof1.hpp"
#include "lemma34_proof2.hpp"
#include "lemma34_proof3.hpp"

static void run_lemma34_proof1() {
    const auto printer = ivarp::critical_printer(std::cerr, lemma34_proof1::system,
                                                 printable_expression("S", lemma34_proof1::S),
                                                 printable_expression("r(s_1)", r(lemma34_proof1::s1)),
                                                 printable_expression("z", lemma34_proof1::z),
                                                 printable_expression("w_1", lemma34_proof1::w1),
                                                 printable_expression("w_2", lemma34_proof1::w2),
                                                 printable_expression("w_3", lemma34_proof1::w3),
                                                 printable_expression("w_4", lemma34_proof1::w4));
    run_proof("Lemma 34, statement (1)", lemma34_proof1::input, lemma34_proof1::system, printer);
}

static void run_lemma34_proof2() {
    const auto printer = ivarp::critical_printer(std::cerr, lemma34_proof2::system,
                                                 printable_expression("S", lemma34_proof2::S),
                                                 printable_expression("H_3", lemma34_proof2::H3),
                                                 printable_expression("w_1", lemma34_proof2::w1),
                                                 printable_expression("w_2", lemma34_proof2::w2));
    run_proof("Lemma 34, statement (2)", lemma34_proof2::input, lemma34_proof2::system, printer);
}

static void run_lemma34_proof3() {
    const auto printer = ivarp::critical_printer(std::cerr, lemma34_proof3::system);
    run_proof("Lemma 34, statement (3)", lemma34_proof3::input, lemma34_proof3::system, printer);
}

void run_lemma34() {
    run_lemma34_proof1();
    run_lemma34_proof2();
    run_lemma34_proof3();
}
