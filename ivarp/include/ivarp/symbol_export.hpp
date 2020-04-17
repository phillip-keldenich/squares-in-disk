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
// Created by Phillip Keldenich on 09.10.19.
//

#pragma once

// handle export and import of symbols
#if defined(_WIN32) || defined(__WIN32__)

// use declspec(dllexport) on Windows
#ifdef ivarp_EXPORTS // this is defined by cmake when building the shared lib
#define IVARP_EXPORTED_SYMBOL __declspec(dllexport)
#else
#define IVARP_EXPORTED_SYMBOL __declspec(dllimport)
#endif

#elif defined(__GNUC__) && __GNUC__ >= 4

// for reasonable GCC/clang versions on Unix, use the visibility attribute
#define IVARP_EXPORTED_SYMBOL  __attribute__((__visibility__ ("default")))

#else
#define IVARP_EXPORTED_SYMBOL
#endif
