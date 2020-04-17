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
// Created by Phillip Keldenich on 17.02.20.
//

#pragma once
#include "ivarp/metaprogramming.hpp"
#include <cstddef>
#include <algorithm>
#include <boost/config.hpp>

namespace ivarp {
    /**
     * @brief Combines a bitset with a fixed-size ring-buffer queue.
     * @tparam NumIndices
     */
    template<std::size_t NumIndices> class IndexQueueSet {
    public:
        IndexQueueSet() noexcept : // NOLINT
            m_begin(0), m_end(0)
        {
            std::fill_n(m_present, NumIndices, false);
        }

        template<std::size_t... Elements>
            void enqueue_all(IndexPack<Elements...>)
        {
            std::size_t arr[] = {Elements...};
            for(std::size_t a : arr) {
                enqueue(a);
            }
        }

        template<typename ForwardIt>
            void enqueue_range(ForwardIt begin, ForwardIt end)
        {
            for(; begin != end; ++begin) {
                enqueue(*begin);
            }
        }

        bool empty() const noexcept {
            return m_begin == m_end;
        }

        std::size_t size() const noexcept {
            return (m_begin < m_end) ? (m_end - m_begin) : (NumIndices - m_begin + m_end);
        }

        bool enqueue(std::size_t element) noexcept {
            if(m_present[element]) {
                return false;
            }
            m_present[element] = true;

            m_elements[m_end] = element;
            if(BOOST_UNLIKELY(++m_end >= NumIndices)) {
                m_end = 0;
            }
            return true;
        }

        std::size_t dequeue() noexcept {
            std::size_t element = m_elements[m_begin];
            m_present[element] = false;
            if(BOOST_UNLIKELY(++m_begin >= NumIndices)) {
                m_begin = 0;
            }
            return element;
        }

        bool is_present(std::size_t e) const noexcept {
            return m_present[e];
        }

    private:
        std::size_t m_begin, m_end;
        std::size_t m_elements[NumIndices];
        bool m_present[NumIndices];
    };
}
