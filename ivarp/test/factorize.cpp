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
// Created by Phillip Keldenich on 28.02.20.
//

#include "test_util.hpp"
#include "ivarp/number.hpp"

namespace {
TEST_CASE("[ivarp][number] Factorization") {
    using namespace ivarp;
    using FZ = std::vector<FactorEntry>;

    REQUIRE(factorize(2) == FZ{{2,1}});
    REQUIRE(factorize(3) == FZ{{3,1}});
    REQUIRE(factorize(4) == FZ{{2,2}});
    REQUIRE(factorize(5) == FZ{{5,1}});
    REQUIRE(factorize(6) == FZ{{2,1},{3,1}});
    REQUIRE(factorize(7) == FZ{{7,1}});
    REQUIRE(factorize(8) == FZ{{2,3}});
    REQUIRE(factorize(9) == FZ{{3,2}});
    REQUIRE(factorize(5041) == FZ{{71,2}});
    REQUIRE(factorize(65520) == FZ{{2,4},{3,2},{5,1},{7,1},{13,1}});
    REQUIRE(factorize(526715280) == FZ{{2,4},{3,2},{5,1},{7,1},{13,1},{8039,1}});
    REQUIRE(factorize(65521u*65521u) == FZ{{65521,2}});
    REQUIRE(factorize(65521u*65537u) == FZ{{65521,1},{65537,1}});
    REQUIRE(factorize(4294967291u) == FZ{{4294967291u, 1}});
}
}
