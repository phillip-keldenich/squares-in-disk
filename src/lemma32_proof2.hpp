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
// Created by me on 26.11.19.
//

#pragma once

#include <ivarp/variable_description.hpp>

namespace lemma32_proof2 {
	using namespace ivarp;
	using namespace ivarp::args;

	using lemma32_proof1::s1;
	using lemma32_proof1::h1;
	using lemma32_proof1::h2;
	using lemma32_proof1::z;
	using lemma32_proof1::w1;
	using lemma32_proof1::w2;

	const auto m = maximum(r(s1),z);
	const auto S = square(s1) + A_c(T_inv(s1), h1, w1, h2) + A_11(h2,w2,m) + square(m);
	const auto s1d = variable(s1, "s_1", 0.295_X, 1.3_X);
	const auto h1d = variable(h1, "h_1", 0_Z, s1);
	const auto h2d = variable(h2, "h_2", 0_Z, h1);
	const auto system = constraint_system(s1d, h1d, h2d,
        S <= 1.6_X,
        z <= h2, z <= h1, z <= s1,
		h2 <= h1, h2 <= s1,
		h1 <= s1, r(s1) <= h2, r(s1) <= h1, r(s1) <= s1
	);
	const auto input = prover_input<CTX, lemma32_proof1::VarSplit>(system);
}
