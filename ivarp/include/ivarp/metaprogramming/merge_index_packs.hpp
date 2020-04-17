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
// Created by Phillip Keldenich on 14.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Implement mergesort-like merging of index packs.
    template<typename Curr, typename IP1, typename IP2> struct MergeIndexPacksImpl;

    /// Merge with P2 or both packs empty.
    template<std::size_t... Curr, std::size_t... S1>
        struct MergeIndexPacksImpl<IndexPack<Curr...>, IndexPack<S1...>, IndexPack<>>
    {
        using Type = IndexPack<Curr..., S1...>;
    };

    /// Merge with P1 empty, P2 nonempty.
    template<std::size_t... Curr, std::size_t S2_1, std::size_t... S2>
        struct MergeIndexPacksImpl<IndexPack<Curr...>, IndexPack<>, IndexPack<S2_1, S2...>>
    {
        using Type = IndexPack<Curr..., S2_1, S2...>;
    };

    /// Merge with both non-empty.
    template<std::size_t... Curr, std::size_t S1_1, std::size_t... S1, std::size_t S2_1, std::size_t... S2>
        struct MergeIndexPacksImpl<IndexPack<Curr...>, IndexPack<S1_1, S1...>, IndexPack<S2_1, S2...>>
    {
    private:
        struct S1Less {
            template<typename = void> struct Lazy {
                using Type = typename
                    MergeIndexPacksImpl<IndexPack<Curr..., S1_1>, IndexPack<S1...>, IndexPack<S2_1,S2...>>::Type;
            };
        };
        struct S2Less {
            template<typename = void> struct Lazy {
                using Type = typename
                    MergeIndexPacksImpl<IndexPack<Curr..., S2_1>, IndexPack<S1_1,S1...>, IndexPack<S2...>>::Type;
            };
        };
        struct Equal {
            template<typename = void> struct Lazy {
                using Type = typename
                    MergeIndexPacksImpl<IndexPack<Curr..., S1_1>, IndexPack<S1...>, IndexPack<S2...>>::Type;
            };
        };
        using NextSelector = std::conditional_t<(S1_1 < S2_1), S1Less,
                             std::conditional_t<(S2_1 < S1_1), S2Less, Equal>>;

    public:
        using Type = typename NextSelector::template Lazy<>::Type;
    };
}

    /// Merge two sorted index packs, each with unique elements into a single sorted index pack with unique elements.
    template<typename IndexPack1, typename IndexPack2> using MergeIndexPacks = typename
        impl::MergeIndexPacksImpl<IndexPack<>, IndexPack1, IndexPack2>::Type;
}
