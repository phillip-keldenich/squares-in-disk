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
// Created by Phillip Keldenich on 25.11.19.
//

#pragma once

#include <ivarp/math_fn.hpp>

using CTX = ivarp::DefaultContextWithNumberType<ivarp::IDouble>;

namespace {
    using namespace ivarp;
    using namespace ivarp::args;

    const auto T = (2_Z / sqrt(ensure_expr(5_Z))) * sqrt(maximum(0_Z, 1_Z - square(x0)/5_Z)) - 0.8_X * x0;
    const auto T_inv = sqrt(maximum(1_Z - square(x0)/4_Z, 0_Z)) - x0;

    const auto r = 0.25_X * (-x0 - 2_Z*T_inv(x0) + sqrt(8_Z - square(x0 - 2_Z*T_inv(x0))));

    const auto A_1_case1 = 0.5_X*x0*x1 + 0.25_X*square(x0);
    const auto A_1_case2 = square(x0) + (x1 - x0 - x2) * x2;
    const auto A_1_case3 = 0.5_X * x0 * (1_Z + x1) - square(x2);
    const auto A_1 = maximum(A_1_case1, A_1_case2, A_1_case3);

    const auto A_11_case1 = square(x0);
    const auto A_11_case2 = square(x0) + square(x2);
    const auto A_11_case3 = A_1(x0,x1,x2);
    const auto A_11 = if_then_else(x1 < x0 + x2, A_11_case1,
                      if_then_else(x1 <= 2_Z*x0, A_11_case2, A_11_case3));

    const auto w = 2_Z * sqrt(maximum(0_Z, minimum(1_Z - square(x0), 1_Z - square(x0-x1))));
    const auto X/*(a_i,b_i,u)*/ = if_then_else(
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
    const auto Y/*(a,h_i,w_i,h_{i+1})*/ = 0.5_X * x2 - x1 + X(x0, x0 - x1, x3);

    const auto Y_lev = Y(x0,x1,x2,x3);
    const auto A2/*(lev,hi,wi,h_{i+1])*/ = square(x1) + maximum(Y_lev * x3, minimum(square(Y_lev), 2_Z*square(x3)));
    const auto A_mod/*(lev,h1,w1,h2)*/ = square(x1) + minimum(square(maximum(0_Z, Y_lev)), 2_Z*square(x3));
    const auto A_c/*(c,hi,wi,h_{i+1})*/ = if_then_else(x2 <= 2_Z*x1, A_mod, maximum(A_11(x1,x2,x3), A2));

    const auto G = x0 * sqrt(1_Z-square(x0)) + asin(x0);
    const auto est = G(maximum(x1 - x0 - 1_Z, -1_Z)) - G(ensure_expr(-1_Z)) + 2*square(x0) - x0*x1;

    const auto Ab_case1/*(h_j,w_j,h_{j+1})*/ = maximum(
        0.5_X*x0*x1 + 0.25_X * square(x0),
        square(x0) + (x1-x0-x2) * x2
    );
    const auto Ab_case2 = 0.5_X * x0 * x1 + 0.25_X*square(x1);
    const auto Ab_case3 = square(x0);
    const auto Ab = if_then_else(x1 >= 3_Z*x0, Ab_case1,
                                 if_then_else(x1 < 2_Z*x0, Ab_case3, Ab_case2));


    const auto ABH/*(H,W,h_{j+1})*/ = if_then_else(x1 >= 2_Z*x0,
                                                   0.5_X * x0 * x1 + 0.25_X * square(x2),
                                                   0.5_X * x0 * x1);
}
