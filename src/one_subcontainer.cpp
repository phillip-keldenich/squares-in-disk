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

#include "proof_auxiliaries.hpp"
#include "B4.hpp"

namespace one_subcontainer_proof1 {
    using namespace ivarp;
    using namespace ivarp::args;
	using B_functions::B_4;
	using namespace aux_functions;
	using CTX = DefaultContextWithNumberType<IDouble>;

    static const auto s1 = x0;
    static const auto h1 = x1;
	static const auto sn = x2;

    static const auto z = T(-T_inv(s1) + h1);
    static const auto w1 = w(T_inv(s1),h1);
	static const auto F_3 = 
		square(s1) + B_4(T_inv(s1), h1, w1, sn) + square(sn) + last_square(s1, sn);

	static const auto not_case_2 =
		(s1 > 1_Z / sqrt(ensure_expr(2_Z))) ||
		w1 < 2_Z * h1 ||
		square(s1) + square(h1) + 2_Z * square(sn) <= 1.56_X;

    const auto system = constraint_system(
		variable(s1, "s_1", 0.295_X, 1.3_X), variable(h1, "h_1", 0_Z, s1), variable(sn, "s_n", z, h1),
        F_3 <= 1.6_X, s1 >= h1, h1 >= sn, z >= 0_Z, h1 <= T_inv(s1) + 1_Z, not_case_2
	);
    const auto input = prover_input<CTX, U64Pack<dynamic_subdivision(128, 8), 256, 256>>(system);
}

static void run_one_subcontainer_proof1() {
	using namespace one_subcontainer_proof1;
	using namespace aux_functions;
	using namespace B_functions;


    const auto printer = ivarp::critical_printer(std::cerr, one_subcontainer_proof1::system,
                                                 printable_expression("z", z),
                                                 printable_expression("F_3", F_3),
                                                 printable_expression("sigma(s_1)", sigma(s1)),
												 printable_expression("w_1", w1),
												 printable_expression("B_4", B_4(T_inv(s1), h1, w1, sn)));
    run_proof("One Subcontainer Lemma", input, one_subcontainer_proof1::system, printer);
}

void run_one_subcontainer() {
    run_one_subcontainer_proof1();
}

