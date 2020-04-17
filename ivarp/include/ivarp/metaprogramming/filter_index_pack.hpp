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
// Created by Phillip Keldenich on 30.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<template<std::size_t> class Predicate, std::size_t... Inds> struct FilterIndexPackImplI {
        using Type = IndexPack<>;
    };

    template<template<std::size_t> class Predicate, std::size_t I1, std::size_t... Inds>
        struct FilterIndexPackImplI<Predicate, I1, Inds...>
    {
    private:
        struct PredTrue {
            template<typename = void> struct Lazy {
                using Type = typename FilterIndexPackImplI<Predicate, Inds...>::Type::template Prepend<I1>::Type;
            };
        };
        struct PredFalse {
            template<typename = void> struct Lazy {
                using Type = typename FilterIndexPackImplI<Predicate, Inds...>::Type;
            };
        };

    public:
        using Type = typename std::conditional_t<Predicate<I1>::value, PredTrue, PredFalse>::template Lazy<>::Type;
    };

    template<template<std::size_t> class Predicate, typename IndPack> struct FilterIndexPackImpl;
    template<template<std::size_t> class Predicate, std::size_t... Inds>
        struct FilterIndexPackImpl<Predicate, IndexPack<Inds...>> : FilterIndexPackImplI<Predicate, Inds...>
    {};
}

    /**
     * Filter the given index pack according to the given predicate, i.e., produce an IndexPack
     * containing all indices I in IndPack for which Predicate<I>::value is true.
     */
    template<template<std::size_t> class Predicate, typename IndPack> using FilterIndexPack =
        typename impl::FilterIndexPackImpl<Predicate,IndPack>::Type;
}
