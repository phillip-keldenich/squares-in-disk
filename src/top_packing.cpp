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
#include "top_packing.hpp"

void run_top_packing() {
    const auto printer1 = ivarp::critical_printer(std::cerr, top_packing::proof1::system,
                                                  printable_expression("D", top_packing::proof1::D));

    run_proof("Top Packing Lemma, statement (1)", top_packing::proof1::input, top_packing::proof1::system, printer1);

    const auto printer2 = ivarp::critical_printer(std::cerr, top_packing::proof2::system,
                                                  printable_expression("D", top_packing::proof2::D));
    run_proof("Top Packing Lemma, statement (2)", top_packing::proof2::input, top_packing::proof2::system, printer2);
}

