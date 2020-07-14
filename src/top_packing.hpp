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
// Created by Phillip Keldenich on 29.11.19.
//

#pragma once

#include "fn_aux.hpp"
#include <ivarp/prover.hpp>

namespace top_packing {
    using namespace ivarp;
	using aux_functions::T_inv;
	using aux_functions::sigma;
	using aux_functions::s1_star;
	using CTX = DefaultContextWithNumberType<IDouble>;

    const auto s1 = args::x0;
	const auto l1 = sqrt(1_Z - square(T_inv(s1))) - 0.5_X * s1;

    const auto sigma_2sq2 = (1_Z / (2_Z*sqrt(ensure_expr(2_Z)))) * sigma(s1);
    const auto s1d = variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X)));

    namespace proof1 {
		const auto xp1 = 0.5_X * s1 + sigma_2sq2;
		const auto yp1 = if_then_else(s1 <= s1_star, T_inv(s1) + sigma(s1) + sigma_2sq2, 0.5_X * sigma + sigma_2sq2);
		const auto F_TP1 = square(xp1) + square(yp1);

        const auto system = constraint_system(s1d, s1 >= l1, F_TP1 > 1_Z);
        const auto input = prover_input<CTX, U64Pack<dynamic_subdivision(128,8)>>(system);
    }

    namespace proof2 {
		const auto xp2 = 0.5_X * s1 + 0.645_X * sigma(s1);
		const auto yp2 = if_then_else(s1 <= s1_star, T_inv(s1) + 2_Z * 0.645_X * sigma(s1), 0.79_X * sigma(s1));
        const auto F_TP2 = square(xp2) + square(yp2);

        const auto system = constraint_system(s1d, s1 >= l1, F_TP2 > 1_Z);
        const auto input = prover_input<CTX, U64Pack<dynamic_subdivision(128,8)>>(system);
    }
}

