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
// Created by Phillip Keldenich on 13.11.19.
//

#pragma once

#include "ivarp/cuda.hpp"

namespace ivarp {
    enum class BoundDirection : unsigned {
        NONE = 0, ///< The result of the bound function is not necessarily any bound (failure to reformulate)
        LEQ  = 1, ///< The result of the bound function is an upper bound
        GEQ  = 2, ///< The result of the bound function is a lower bound
        BOTH = 3  ///< The result of the bound function is both a lower and an upper bound
    };

    static inline IVARP_H std::ostream& operator<<(std::ostream& o, BoundDirection bd) {
        using CCP = const char*;
        CCP names[] = {
            "NONE", "LEQ", "GEQ", "BOTH"
        };
        return o << names[static_cast<unsigned>(bd)];
    }

    static inline IVARP_HD constexpr BoundDirection operator|(BoundDirection b1, BoundDirection b2) noexcept {
        return static_cast<BoundDirection>(static_cast<unsigned>(b1) | static_cast<unsigned>(b2));
    }

    static inline IVARP_HD constexpr BoundDirection operator&(BoundDirection b1, BoundDirection b2) noexcept {
        return static_cast<BoundDirection>(static_cast<unsigned>(b1) & static_cast<unsigned>(b2));
    }

    static inline IVARP_HD constexpr BoundDirection operator^(BoundDirection b1, BoundDirection b2) noexcept {
        return static_cast<BoundDirection>(static_cast<unsigned>(b1) ^ static_cast<unsigned>(b2));
    }

    static IVARP_HD inline BoundDirection& operator|=(BoundDirection& b1, BoundDirection b2) noexcept {
        b1 = b1 | b2;
        return b1;
    }

    static IVARP_HD inline BoundDirection& operator&=(BoundDirection& b1, BoundDirection b2) noexcept {
        b1 = b1 & b2;
        return b1;
    }

    static IVARP_HD inline BoundDirection& operator^=(BoundDirection& b1, BoundDirection b2) noexcept {
        b1 = b1 ^ b2;
        return b1;
    }

    static IVARP_HD inline constexpr BoundDirection operator~(BoundDirection b1) noexcept {
        return static_cast<BoundDirection>(static_cast<unsigned>(b1) ^ static_cast<unsigned>(BoundDirection::BOTH));
    }
}
