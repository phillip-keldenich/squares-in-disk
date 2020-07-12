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
// Created by Phillip Keldenich on 09.07.2020.
//

/// @file see https://gitlab.ibr.cs.tu-bs.de/alg/ivarp/issues/10

#include "ivarp/run_prover.hpp"
#include "ivarp/critical_printer.hpp"
#include "../test_util.hpp"

namespace {
	struct RequireHandler {
		template<typename Context, typename Critical> void operator()(const Context&, const Critical& crits) const {
			std::stringstream buf;
			buf << "Error: Critical cuboid: s_1: " << crits[0] << ", s_2: " << crits[1] << ", s_3: " << crits[2] << std::endl;
			std::cerr << buf.rdbuf();
			REQUIRE(false);
		}
	};

	static inline void run_one_subcontainer_proof1() {
		using namespace ivarp;
		using namespace ivarp::args;

		const auto B_1 = maximum(
			0.5_X * x0 * x1 + 0.25_X * square(x0),
			square(x0) + (x1 - x0 - x2) * x2,
			0.5_X * x0 * (x1 + x0) - square(x2)
		);

		const auto B_2 = if_then_else(
		x1 < x0 + x2, square(x0),
		if_then_else(x1 < 2_Z * x0, square(x0) + square(x2), B_1(x0,x1,x2))
		);

		const auto T =
			(2_Z / sqrt(ensure_expr(5_Z))) * sqrt(maximum(0_Z, 1_Z - square(x0)/5_Z)) - 0.8_X * x0;

		const auto T_inv =
			sqrt(maximum(1_Z - 0.25_X * square(x0), 0_Z)) - x0;

		const auto w =
			2_Z * sqrt(maximum(0_Z, minimum(1_Z - square(x0), 1_Z - square(x0-x1))));

		const auto sigma = if_then_else(x0 <= sqrt((2_Z + sqrt(ensure_expr(2_Z))) / 3_Z),
			0.25_X * (-x0 - 2_Z*T_inv(x0) + sqrt(8_Z - square(x0 - 2_Z*T_inv(x0)))),
			0.2_X * (sqrt(20_Z - square(x0)) - 2_Z * x0)
		);

		const auto last_square = if_then_else(
			sigma(x0) >= x1, 0.83_X * square(sigma(x0)), 0_Z
		);

		const auto X = if_then_else(
			x1 >= 0_Z,
			sqrt(1_Z - square(x1 + x2)) - x2,
			if_then_else(
				x0 < 0_Z,
				sqrt(maximum(0_Z, 1_Z - square(x0 + x2))) - x2,
				if_then_else(
					x2 <= 2_Z * minimum(x0,-x1),
					T_inv(x2),
					sqrt(maximum(0_Z, 1_Z - square(x2 - minimum(x0,-x1)))) - x2
				)
			)
		);

		const auto Y = 0.5_X * x2 - x1 + X(x0, x0 - x1, x3);

		const auto B_3 = square(x1) + maximum(
			maximum(0_Z, Y(x0,x1,x2,x3) * x3),
			minimum(square(maximum(0_Z,Y(x0,x1,x2,x3))), 2_Z * square(x3))
		);

		const auto B_4 = maximum(B_2(x1, x2, x3), B_3(x0,x1,x2,x3));
		const auto s1 = x0;
		const auto h1 = x1;
		const auto sn = x2;
		const auto z = T(-T_inv(s1) + h1);
		const auto w1 = w(T_inv(s1),h1);
		const auto F_OC = square(s1) + B_4(T_inv(s1), h1, w1, sn) + square(sn) + last_square(s1, sn);
		const auto not_case_2 = (s1 > 1_Z / sqrt(ensure_expr(2_Z))) || w1 < 2_Z * h1 || square(s1) + square(h1) + 2_Z * square(sn) <= 1.56_X;
		const auto csystem = constraint_system(
			variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X))), variable(h1, "h_1", 0_Z, s1), variable(sn, "s_n", maximum(0_Z, z), h1),
			F_OC <= 1.6_X, s1 >= h1, h1 >= sn, z >= 0_Z, z <= sn, sn >= z, h1 <= T_inv(s1) + 1_Z, not_case_2
		);
		using CTX = DefaultContextWithNumberType<IDouble>;
		const auto input = prover_input<CTX, U64Pack<dynamic_subdivision(128, 8), 256, 256>>(csystem);
		const auto printer = ivarp::critical_printer(std::cerr, csystem,
													 printable_expression("z", z),
													 printable_expression("F_OC", F_OC),
													 printable_expression("sigma(s_1)", sigma(s1)),
													 printable_expression("w_1", w1),
													 printable_expression("B_4", B_4(T_inv(s1), h1, w1, sn)));
		ivarp::ProofInformation info;
		ivarp::ProverSettings settings;
		settings.max_iterations_per_node = 1;
		REQUIRE(run_prover(input, RequireHandler{}, &info, settings));
	}
}

