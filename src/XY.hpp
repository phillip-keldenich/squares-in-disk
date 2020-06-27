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

#include "fn_aux.hpp"

namespace aux_functions {
	static const auto X/*(a_i,b_i,u)*/ = if_then_else(
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

	static const auto Y/*(a,h_i,w_i,h_{i+1})*/ = 0.5_X * x2 - x1 + X(x0, x0 - x1, x3);
}

