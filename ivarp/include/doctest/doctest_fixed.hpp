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
// Created by Phillip Keldenich on 11.12.19.
//

#pragma once

// doctest does really weird (and technically illegal/undefined-behavior) stuff (forward-declaring things in namespace std), which does arcanely broken stuff w.r.t. linking.
// this is probably due to inline namespaces, cxx11-abi fixes, etc. - this is why you really _should not_ rely on undefined behavior just because it "works for you".
// if GCC breaks down during linking, e.g., only for debug or relwithdebug, this may be the reason.
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#define DOCTEST_CONFIG_USE_STD_HEADERS 1
#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS 1

#include "doctest.h"
