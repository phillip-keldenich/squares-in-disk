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

namespace lemma34_proof2 {
    using lemma34_proof1::s1;
    using lemma34_proof1::s1d;
    using lemma34_proof1::h1;
    using lemma34_proof1::h1d;
    using lemma34_proof1::h2;
    using lemma34_proof1::h2d;
    using lemma34_proof1::h3;
    using lemma34_proof1::h3d;
    using lemma34_proof1::w1;
    using lemma34_proof1::w2;

    using VarSplit = U64Pack<dynamic_subdivision(128, 4), dynamic_subdivision(64, 2), dynamic_subdivision(32, 2), 256>;

    const auto H3 = T_inv(s1) - h1 - h2 + 1;
    const auto S = square(s1) + 0.83_X * square(r(s1)) + A_c(T_inv(s1), h1, w1, h2)
                   + A_c(T_inv(s1) - h1, h2, w2, h3) + est(h3, H3);

    const auto system = constraint_system(s1d, h1d, h2d, h3d,
        S <= 1.6_X, H3 <= 1_Z, h2 >= T_inv(s1) - h1, h1 >= T_inv(s1) - h2
    );

    const auto input = prover_input<CTX, VarSplit>(system);
}
