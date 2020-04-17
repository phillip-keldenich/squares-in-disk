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
namespace impl {
    template<std::size_t Begin, std::size_t End> struct IndexRangeImpl;
}
    /// Create a contiguous range of indices [Begin,End).
    template<std::size_t Begin, std::size_t End> using IndexRange = typename impl::IndexRangeImpl<Begin,End>::Type;
}

namespace ivarp {
namespace impl {
    /// Handle the common cases directly.
    template<> struct IndexRangeImpl<0u,0u> {
        using Type = IndexPack<>;
    };
    template<> struct IndexRangeImpl<0u,1u> {
        using Type = IndexPack<0u>;
    };
    template<> struct IndexRangeImpl<0u,2u> {
        using Type = IndexPack<0u, 1u>;
    };
    template<> struct IndexRangeImpl<0u,3u> {
        using Type = IndexPack<0u, 1u, 2u>;
    };
    template<> struct IndexRangeImpl<0u,4u> {
        using Type = IndexPack<0u, 1u, 2u, 3u>;
    };
    template<> struct IndexRangeImpl<0u,5u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u>;
    };
    template<> struct IndexRangeImpl<0u,6u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u>;
    };
    template<> struct IndexRangeImpl<0u,7u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u>;
    };
    template<> struct IndexRangeImpl<0u,8u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u>;
    };
    template<> struct IndexRangeImpl<0u,9u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u>;
    };
    template<> struct IndexRangeImpl<0u,10u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u>;
    };
    template<> struct IndexRangeImpl<0u,11u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u>;
    };
    template<> struct IndexRangeImpl<0u,12u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u>;
    };
    template<> struct IndexRangeImpl<0u,13u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u>;
    };
    template<> struct IndexRangeImpl<0u,14u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u>;
    };
    template<> struct IndexRangeImpl<0u,15u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u>;
    };
    template<> struct IndexRangeImpl<0u,16u> {
        using Type = IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u>;
    };

    /// On the off chance that we need more than 16 elements.
    template<std::size_t End> struct IndexRangeImpl<0u,End> {
        using Type = IndexConcat<
            IndexPack<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u>,
            AddOffset<IndexRange<0u, End-16u>, 16u>
        >;
    };

    /// For other starting points than 0.
    template<std::size_t Begin, std::size_t End> struct IndexRangeImpl {
        using Type = AddOffset<IndexRange<0u, (End > Begin ? End-Begin : 0u)>, Begin>;
    };
}
}
