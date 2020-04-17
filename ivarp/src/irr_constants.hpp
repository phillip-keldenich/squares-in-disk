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
// Created by Phillip Keldenich on 10.10.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename FloatType> struct FloatConstants;
    template<> struct FloatConstants<float> {
        static const IFloat pi;         // pi constant
        static const IFloat two_pi;     // 2*pi
        static const IFloat rec_two_pi; // 1/(2*pi)
    };

    template<> struct FloatConstants<double> {
        static const IDouble pi;         // pi constant
        static const IDouble two_pi;     // 2*pi
        static const IDouble rec_two_pi; // 1/(2*pi)
    };

    template<> struct FloatConstants<Rational> {
        static const IRational pi;         // pi constant
        static const IRational two_pi;     // 2*pi
        static const IRational rec_two_pi; // 1/(2*pi)
    };

    IRational rational_rec_two_pi(unsigned precision);
}
}
