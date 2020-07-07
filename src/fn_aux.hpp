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
// Created by Phillip Keldenich on 26.06.2020.
//

#pragma once

#include <ivarp/math_fn.hpp>

namespace aux_functions {
	using namespace ivarp;
	using namespace ivarp::args;

	static const auto T/*(u)*/ =
		(2_Z / sqrt(ensure_expr(5_Z))) * sqrt(maximum(0_Z, 1_Z - square(x0)/5_Z)) - 0.8_X * x0;

    static const auto T_inv/*(s)*/ =
		sqrt(maximum(1_Z - 0.25_X * square(x0), 0_Z)) - x0;

	static const auto w/*(y_t,h)*/ =
		2_Z * sqrt(maximum(0_Z, minimum(1_Z - square(x0), 1_Z - square(x0-x1))));

	static const auto sigma/*(s_1)*/ = if_then_else(x0 <= sqrt((2_Z + sqrt(ensure_expr(2_Z))) / 3_Z),
		0.25_X * (-x0 - 2_Z*T_inv(x0) + sqrt(8_Z - square(x0 - 2_Z*T_inv(x0)))),
		0.2_X * (sqrt(20_Z - square(x0)) - 2_Z * x0)
	);

	// this is a shorthand for {0 if s_n > sigma else 0.83sigma^2}
	static const auto last_square/*(s1,sn)*/ = if_then_else(
		sigma(x0) >= x1, 0.83_X * square(sigma(x0)), 0_Z
	);
}

