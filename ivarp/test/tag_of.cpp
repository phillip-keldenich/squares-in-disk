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
// Created by Phillip Keldenich on 08.01.20.
//

#include "ivarp/math_fn.hpp"

using namespace ivarp;
using namespace ivarp::args;

static_assert(std::is_same<TagOf<decltype(x0 + x1)>, MathOperatorTagAdd>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 - x1)>, MathOperatorTagSub>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 * x1)>, MathOperatorTagMul>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 / x1)>, MathOperatorTagDiv>::value, "Wrong tag!");

static_assert(std::is_same<TagOf<decltype(x0 < x1)>, BinaryMathPredLT>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 <= x1)>, BinaryMathPredLEQ>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 > x1)>, BinaryMathPredGT>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 >= x1)>, BinaryMathPredGEQ>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 == x1)>, BinaryMathPredEQ>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 != x1)>, BinaryMathPredNEQ>::value, "Wrong tag!");

static_assert(std::is_same<TagOf<decltype(x0 < x1 || x0 == x1)>, MathPredOrSeq>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(x0 < x1 && x0 == x1)>, MathPredAndSeq>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype((x0 < x1) | (x0 == x1))>, MathPredOr>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype((x0 < x1) & (x0 == x1))>, MathPredAnd>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype((x0 < x1) ^ (x0 == x1))>, MathPredXor>::value, "Wrong tag!");

static_assert(std::is_same<TagOf<decltype(-x0)>, MathOperatorTagUnaryMinus>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(sin(x0))>, MathSinTag>::value, "Wrong tag!");

static_assert(std::is_same<TagOf<decltype(if_then_else(x0 < x1, x1, x0))>, MathTernaryIfThenElse>::value, "Wrong tag!");
static_assert(std::is_same<TagOf<decltype(maximum(x0,x1))>, MathNAryMaxTag>::value, "Wrong tag!");

static_assert(!HasTag<decltype(x0)>::value, "Arguments do not have tags!");
static_assert(!HasTag<decltype(constant(15_Z))>::value, "Constants do not have tags!");
