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

namespace five_subcontainers_proof1 {
	// The case for five subtainers and s_n > sigma.
	using namespace ivarp;
	using namespace ivarp::args;
	using namespace aux_functions;
	using B_functions::B_4;
	using CTX = DefaultContextWithNumberType<IDouble>;

	static const auto s1 = x0;
	static const auto sn = x1;
	static const auto h1 = x2;
	static const auto h2 = x3;
	static const auto h3 = x4;
	static const auto h4 = x5;
	static const auto h5 = x6;
	static const auto z = T(-T_inv(s1) + h1 + h2 + h3 + h4 + h5);
	static const auto w1 = w(T_inv(s1), h1);
	static const auto w2 = w(T_inv(s1) - h1, h2);
	static const auto w3 = w(T_inv(s1) - h1 - h2, h3);
	static const auto w4 = w(T_inv(s1) - h1 - h2 - h3, h4);
	static const auto w5 = w(T_inv(s1) - h1 - h2 - h3 - h4, h5);

	static const auto F_5C_1 = square(s1) + square(sn) +
				               B_4(T_inv(s1), h1, w1, h2) +
							   B_4(T_inv(s1)-h1, h2, w2, h3) + 
                               B_4(T_inv(s1)-h1-h2, h3, w3, h4) +
							   B_4(T_inv(s1)-h1-h2-h3, h4, w4, h5) +
							   B_4(T_inv(s1)-h1-h2-h3-h4, h5, w5, sn);

	static const auto system = constraint_system(
		variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X))),
		variable(sn, "s_n", sigma(s1), s1),
		variable(h1, "h_1", sn, s1),
		variable(h2, "h_2", sn, h1),
		variable(h3, "h_3", sn, h2),
		variable(h4, "h_4", sn, h3),
		variable(h5, "h_5", sn, h4),
		sn > z, z >= 0_Z, F_5C_1 <= 1.6_X,
		h1 <= T_inv(s1) + 1_Z - 4_Z * sigma(s1),
		h2 <= T_inv(s1) + 1_Z - h1 - 3_Z * sigma(s1),
		h3 <= T_inv(s1) + 1_Z - h1 - h2 - 2_Z * sigma(s1),
		h4 <= T_inv(s1) + 1_Z - h1 - h2 - h3 - sigma(s1),
		h5 <= T_inv(s1) + 1_Z - h1 - h2 - h3 - h4
	);

	static const auto input = prover_input<
		CTX, U64Pack<
			dynamic_subdivision(128, 4), 
			dynamic_subdivision(16, 4),
			dynamic_subdivision(16, 4),
			16, 16, 16, 4>>(system);
}

static void run_five_subcontainers_proof1() {
	using namespace five_subcontainers_proof1;
	using namespace aux_functions;
	using namespace B_functions;

    const auto printer = ivarp::critical_printer(std::cerr, five_subcontainers_proof1::system,
                                                 printable_expression("z", z),
                                                 printable_expression("F_5C_1", F_5C_1));
	run_proof("Five Subcontainers, s_n > sigma", input, five_subcontainers_proof1::system, printer);
}

void run_five_subcontainers() {
    run_five_subcontainers_proof1();
}
