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
// Created by Phillip Keldenich on 02.07.2020.
//

#include "proof_auxiliaries.hpp"
#include "B4.hpp"

namespace three_subcontainers_proof1 {
	using namespace ivarp;
	using namespace ivarp::args;
	using namespace aux_functions;
	using B_functions::B_4;
	using CTX = DefaultContextWithNumberType<IDouble>;

	static const auto s1 = x0;
	static const auto h1 = x1;
	static const auto h2 = x2;
	static const auto h3 = x3;
	static const auto sn = x4;
	static const auto z = T(-T_inv(s1) + h1 + h2 + h3);
	static const auto w1 = w(T_inv(s1), h1);
	static const auto w2 = w(T_inv(s1) - h1, h2);
	static const auto w3 = w(T_inv(s1) - h1 - h2, h3);
	static const auto F_3C_1 = square(s1) + 
				               B_4(T_inv(s1), h1, w1, h2) + B_4(T_inv(s1)-h1, h2, w2, h3) + 
                               B_4(T_inv(s1)-h1-h2, h3, w3, sn) + square(sn) + last_square(s1, sn);

	static const auto system = constraint_system(
		variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X))),
		variable(h1, "h_1", 0_Z, s1),
		variable(h2, "h_2", 0_Z, h1),
		variable(h3, "h_3", 0_Z, h2),
		variable(sn, "s_n", maximum(0_Z, z), h3),
		z >= 0_Z,
		h1 <= s1, h2 <= h1,
		sn <= h2, h1 <= T_inv(s1) + 1_Z,
		h2 <= T_inv(s1) + 1_Z - h1,
		h3 <= T_inv(s1) + 1_Z - h1 - h2,
		F_3C_1 <= 1.6_X
	);

	static const auto input = prover_input<
		CTX, U64Pack<
			dynamic_subdivision(64, 8), 
			dynamic_subdivision(64, 8),
			dynamic_subdivision(16, 4), 64, 32>>(system);
}

static void run_three_subcontainers_proof1() {
	using namespace three_subcontainers_proof1;
	using namespace aux_functions;
	using namespace B_functions;

    const auto printer = ivarp::critical_printer(std::cerr, three_subcontainers_proof1::system,
                                                 printable_expression("z", z),
                                                 printable_expression("F_3C_1", F_3C_1),
                                                 printable_expression("sigma(s_1)", sigma(s1)));
	run_proof("Three Subcontainer Lemma", input, three_subcontainers_proof1::system, printer);
}

void run_three_subcontainers() {
    run_three_subcontainers_proof1();
}

