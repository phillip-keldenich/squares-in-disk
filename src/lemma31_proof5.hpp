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
// Created by Phillip Keldenich on 26.11.19.
//

#pragma once

namespace lemma31_proof5 {
    using namespace ivarp;
    using namespace ivarp::impl;
    using lemma31_proof1::z;

    const auto s1 = x0;
    const auto h1 = x1;
    const auto S = square(s1) + square(h1) + 3 * square(maximum(r(s1), z));
    using VarSplit = U64Pack<dynamic_subdivision(128, 8), 128>;
    const auto s1d = variable(s1, "s_1", 0.295_X, 1.3_X);
    const auto h1d = variable(h1, "h_1", 0_Z, s1);
    const auto system = constraint_system(s1d, h1d, S <= 1.6_X, z <= h1, z <= s1, h1 <= s1);
    const auto input = prover_input<CTX, VarSplit>(system);
}
