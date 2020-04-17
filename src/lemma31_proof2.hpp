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

namespace lemma31_proof2 {
    using namespace ivarp;
    using namespace ivarp::args;
    using lemma31_proof1::s1;
    using lemma31_proof1::h1;
    using lemma31_proof1::w1;
    const auto sn = x2;
    using lemma31_proof1::z;
    using lemma31_proof1::T_in;

    const auto S = square(s1) + square(sn) + square(h1);
    const auto Y1 = Y(T_in(s1), h1, w1, sn);
    const auto s1d = variable(s1, "s_1", 0.295_X, 1.3_X);
    const auto h1d = variable(h1, "h_1", 0_Z, 1.3_X);
    const auto snd = variable(sn, "s_n", r(s1), 1.3_X);
    const auto system = constraint_system(s1d, h1d, snd,
        S <= 1.6_X, z <= sn, z <= h1, z <= s1, sn <= h1, sn <= s1,
        h1 <= s1, sn >= r(s1), Y1 <= 0_Z, h1 <= 1_Z + T_in(s1));
    using VarSplit = U64Pack<dynamic_subdivision(64, 8), dynamic_subdivision(64, 4), 128>;
    const auto input = prover_input<CTX, VarSplit>(system);
}
