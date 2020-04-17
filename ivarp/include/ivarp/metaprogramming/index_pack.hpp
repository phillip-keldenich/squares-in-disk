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
#include <cstddef>
#include <cstdint>

namespace ivarp {
    /**
     * @brief A type with a constructor that takes (and ignores) any number of arguments of any type.
     *
     * The types must be allowed to be passed, i.e., not void; this is useful to handle a variadic template
     * parameter-based foreach, because the order of evaluation in {}-initialization is fixed (whereas function
     * call parameters can be evaluated in any order).
     */
    struct ConstructWithAny {
        template<typename... Args> IVARP_HD ConstructWithAny(const Args&... /*args*/) noexcept {}
    };

    /// Wrapper around a sequence of std::size_t values.
    template<std::size_t... S> struct IndexPack {
        constexpr IndexPack() noexcept = default;

        static constexpr std::size_t size = sizeof...(S);

        /// At<X>::value is the value of the Xth entry in the sequence.
        template<std::size_t X> struct At;

        /// Prepend an element to the sequence.
        template<std::size_t X> struct Prepend {
            using Type = IndexPack<X,S...>;
        };

        /// Append an element to the sequence.
        template<std::size_t X> struct Append {
            using Type = IndexPack<S...,X>;
        };

        template<typename Callable> static void call_for_each(Callable&& c) {
            // use {} and a constructor instead of () to evaluate in the right order
            ConstructWithAny{(static_cast<void>(c(std::integral_constant<std::size_t, S>{})), false)...};
        }
    };

    template<std::uint64_t... S> struct U64Pack {
        static constexpr std::size_t size = sizeof...(S);
    };
}
