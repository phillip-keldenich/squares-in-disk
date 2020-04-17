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
// Created by Phillip Keldenich on 27.11.19.
//

#pragma once

#include "lemma33_proof1.hpp"

namespace lemma34_proof1 {
    using namespace ivarp;

    using lemma33_proof1::s1;
    using lemma33_proof1::h1;
    using lemma33_proof1::h2;
    using lemma33_proof1::h3;
    const auto h4 = args::x4;
    using VarSplit = U64Pack<dynamic_subdivision(64,4), dynamic_subdivision(64,4), 64, 64, 32>;
    const auto s1d = variable(s1, "s_1", 0.295_X, 1.3_X);
    const auto h1d = variable(h1, "h_1", 0_Z, s1);
    const auto h2d = variable(h2, "h_2", 0_Z, h1);
    const auto h3d = variable(h3, "h_3", 0_Z, h2);
    const auto h4d = variable(h4, "h_4", 0_Z, h3);
    const auto z  = args::x5;
    const auto zd = value(z, T(-T_inv(s1) + h1 + h2 + h3 + h4), "z");

    using lemma33_proof1::w1;
    using lemma33_proof1::w2;
    using lemma33_proof1::w3;
    const auto w4 = w(T_inv(s1) - h1 - h2 - h3, h4);
    const auto wbound = minimum(1_Z - square(args::x0), 1_Z - square(args::x0-args::x1));
    const auto w4bound = wbound(T_inv(s1) - h1 - h2 - h3, h4);
    const auto S = square(s1) + A_11(h1,w1,h2) + A_11(h2,w2,h3) + A_11(h3,w3,h4)
                   + A_11(h4,w4,r(s1)) + square(r(s1));

    const auto system = constraint_system(s1d, h1d, h2d, h3d, h4d, zd,
        S <= 1.6_X,
        z >= 0, z <= h4, z <= h3, z <= h2, z <= h1, z <= s1,
        w4bound >= 0, r(s1) <= h4, r(s1) <= h3, r(s1) <= h2, r(s1) <= h1, r(s1) <= s1
    );

    const auto input = prover_input<CTX,VarSplit>(system);
}
