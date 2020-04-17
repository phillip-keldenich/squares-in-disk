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
// Created by Phillip Keldenich on 2019-10-04.
//

#define DOCTEST_CONFIG_IMPLEMENT

#include <sstream>
#include <iostream>
#include <iomanip>
#include <doctest/doctest_fixed.hpp>
#include "ivarp/rounding.hpp"
#include "ivarp/tuple.hpp"

int main(int argc, char** argv) {
    IVARP_ENSURE_ATLOAD_ROUNDDOWN();
    std::cout << std::setprecision(17);
    std::cerr << std::setprecision(17);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    context.setOption("no-breaks", false);
    int res = context.run();
    if(context.shouldExit()) { // LCOV_EXCL_LINE
        return res; // LCOV_EXCL_LINE
    } // LCOV_EXCL_LINE

    // if we want to do other stuff, we can do it here.
    return res;
}
