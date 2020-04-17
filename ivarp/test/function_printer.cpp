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
// Created by Phillip Keldenich on 19.02.20.
//

#include "ivarp/math_fn.hpp"
#include "test_util.hpp"

namespace {
    using namespace ivarp;
    using namespace ivarp::args;

    TEST_CASE("[ivarp][math_fn][function_printer] Binary") {
        auto fn1 = args::x0 + args::x1 - args::x2 * args::x3;
        require_printable_same(fn1, "x0 + x1 - x2 * x3");

        auto fn2 = args::x0 * (args::x1 + args::x2);
        require_printable_same(fn2, "x0 * (x1 + x2)");

        auto fn3 = args::x0 - (args::x1 + args::x2);
        require_printable_same(fn3, "x0 - (x1 + x2)");

        auto fn4 = args::x0 / (args::x1 * args::x2);
        require_printable_same(fn4, "x0 / (x1 * x2)");

        auto fn5 = args::x0 <= args::x1 * args::x1 + -args::x2;
        require_printable_same(fn5, "x0 <= x1 * x1 + -x2");

        auto fn6 = args::x0 < 1_Z;
        require_printable_same(fn6, "x0 < 1");
    }

    TEST_CASE("[ivarp][math_fn][function_printer] NAry") {
        auto fn1 = maximum(x0,x1,x2,x3);
        require_printable_same(fn1, "max(x0, x1, x2, x3)");

        auto fn2 = minimum(x0,x1,x2,x3);
        require_printable_same(fn2, "min(x0, x1, x2, x3)");

        auto fn3 = if_then_else(x0 <= x1, x2, x3);
        require_printable_same(fn3, "if_then_else(x0 <= x1, x2, x3)");
    }
}
