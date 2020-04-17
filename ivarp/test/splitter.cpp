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
#include <doctest/doctest_fixed.hpp>
#include "ivarp/splitter.hpp"

using namespace ivarp;

TEST_CASE_TEMPLATE("[ivarp][prover] Splitter test", IT, IFloat, IDouble, IRational) {
    IT range{0, 11};
    Splitter<IT> splitter(range, 100);

    REQUIRE(splitter.split_point(0) == 0);
    REQUIRE(splitter.split_point(99) < 11);
    REQUIRE(splitter.split_point(100) == 11);

    IT prev{0};
    int cnt = 0, cnt2 = 0;
    for(IT sub : splitter) {
        if(!definitely(prev == 0)) {
            ++cnt2;
            REQUIRE(prev.ub() == sub.lb());
            REQUIRE(!sub.possibly_undefined());
            REQUIRE(prev.lb() >= 0);
            REQUIRE(sub.ub() <= 11);
        }
        prev = sub;
        ++cnt;
    }

    REQUIRE(cnt == 100);
    REQUIRE(cnt2 == 99);
    REQUIRE(prev.ub() == 11);
}
