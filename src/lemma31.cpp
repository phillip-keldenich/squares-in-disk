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
#include "lemma31_proof1.hpp"
#include "lemma31_proof2.hpp"
#include "lemma31_proof3.hpp"
#include "lemma31_proof4.hpp"
#include "lemma31_proof5.hpp"

static void run_lemma31_proof1() {
    const auto printer = ivarp::critical_printer(std::cerr, lemma31_proof1::system,
                                                 printable_expression("z", lemma31_proof1::z),
                                                 printable_expression("S", lemma31_proof1::S),
                                                 printable_expression("r(s_1)", r(lemma31_proof1::s1)));
    run_proof("Lemma 31, statement (1)", lemma31_proof1::input, lemma31_proof1::system, printer);
}

static void run_lemma31_proof2() {
    const auto printer = ivarp::critical_printer(
        std::cerr, lemma31_proof2::system,
        printable_expression("z", lemma31_proof2::z),
        printable_expression("S", lemma31_proof2::S),
        printable_expression("r(s_1)", r(lemma31_proof2::s1))
    );
    run_proof("Lemma 31, statement (2)", lemma31_proof2::input, lemma31_proof2::system, printer);
}

static void run_lemma31_proof3() {
    const auto printer = ivarp::critical_printer(
        std::cerr, lemma31_proof3::system,
        printable_expression("z", lemma31_proof3::z),
        printable_expression("S", lemma31_proof3::S),
        printable_expression("Y_1", lemma31_proof3::Y1),
        printable_expression("w_1", lemma31_proof3::w1)
    );
    run_proof("Lemma 31, statement (3)", lemma31_proof3::input, lemma31_proof3::system, printer);
}

static void run_lemma31_proof4() {
    const auto printer = ivarp::critical_printer(
        std::cerr, lemma31_proof4::system,
        printable_expression("z", lemma31_proof4::z),
        printable_expression("S", lemma31_proof4::S),
        printable_expression("Y_1", lemma31_proof4::Y1)
    );
    run_proof("Lemma 31, statement (4)", lemma31_proof4::input, lemma31_proof4::system, printer);
}

static void run_lemma31_proof5() {
    const auto printer = ivarp::critical_printer(std::cerr, lemma31_proof5::system);
    run_proof("Lemma 31, statement (5)", lemma31_proof5::input, lemma31_proof5::system, printer);
}

void run_lemma31() {
    run_lemma31_proof1();
    run_lemma31_proof2();
    run_lemma31_proof3();
    run_lemma31_proof4();
    run_lemma31_proof5();
}
