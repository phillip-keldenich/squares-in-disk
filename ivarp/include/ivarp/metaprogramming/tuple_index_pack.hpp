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
// Created by Phillip Keldenich on 06.12.19.
//

#pragma once

namespace ivarp {
    /// Forward-declare Tuple.
    template<typename... Args> struct Tuple;

namespace impl {
    template<typename T> struct TupleIndexPackImpl {
        static_assert(std::is_void<T>::value && !std::is_void<T>::value, "Wrong argument to TupleIndexPack!");
    };
    template<typename... Args> struct TupleIndexPackImpl<std::tuple<Args...>> {
        using Type = IndexRange<0u, sizeof...(Args)>;
    };
    template<typename... Args> struct TupleIndexPackImpl<Tuple<Args...>> {
        using Type = IndexRange<0u, sizeof...(Args)>;
    };
    template<typename T1, typename T2> struct TupleIndexPackImpl<std::pair<T1,T2>> {
        using Type = IndexPack<0u, 1u>;
    };
}

    /// For a n-element tuple, generate IndexPack<0,...,n-1>.
    template<typename TupleType> using TupleIndexPack = typename impl::TupleIndexPackImpl<BareType<TupleType>>::Type;
}
