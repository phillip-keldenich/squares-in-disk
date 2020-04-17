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
// Created by Phillip Keldenich on 11.11.19.
//

#pragma once

#include "ivarp/number.hpp"
#include <boost/iterator/iterator_facade.hpp>
#include <cassert>

namespace ivarp {
    /// Computes split intervals for a given range and a fixed number of subdivisions n.
    /// Valid indices are [0,n).
    template<typename IntervalType> class Splitter {
    public:
        using Interval = IntervalType;
        using Number = typename Interval::NumberType;

        static_assert(IsIntervalType<IntervalType>::value, "Splitter requires an interval type to work on!");

        IVARP_SUPPRESS_HD
        explicit IVARP_HD Splitter(const IntervalType& i, int n) :
            m_range(i), m_n(n), m_ind_width((i.ub() - i.lb()) / n)
        {}

        IVARP_SUPPRESS_HD
        IVARP_HD Number split_point(int i) const {
            if(i >= m_n) {
                return m_range.ub();
            }
            return m_range.lb() + i * m_ind_width;
        }

        IVARP_SUPPRESS_HD
        IVARP_HD IntervalType subrange(int i) const {
            return IntervalType{split_point(i), split_point(i+1)};
        }

        IVARP_HD int size() const noexcept {
            return m_n;
        }

        class Iterator :
            public boost::iterator_facade<Iterator, IntervalType, std::random_access_iterator_tag, IntervalType, int>
        {
        public:
            Iterator() noexcept : m_splitter(nullptr), i(0) {}
            Iterator(const Iterator&) noexcept = default;
            Iterator &operator=(const Iterator&) noexcept = default;

        private:
            explicit Iterator(const Splitter* s, int i) :
                m_splitter(s), i(i)
            {}

            friend class Splitter;
            friend class boost::iterator_core_access;

            IntervalType dereference() const {
                return m_splitter->subrange(i);
            }

            void increment() noexcept {
                ++i;
            }

            void decrement() noexcept {
                --i;
            }

            void advance(int n) noexcept {
                i += n;
            }

            int distance_to(const Iterator& o) const noexcept {
                return o.i - i;
            }

            bool equal(const Iterator& o) const noexcept {
                return i == o.i;
            }

            const Splitter* m_splitter;
            int i;
        };

        Iterator begin() const noexcept {
            return Iterator{this, 0};
        }

        Iterator end() const noexcept {
            return Iterator{this, m_n};
        }

    private:
        IntervalType m_range; ///< The outer range.
        int m_n; ///< The number of subintervals.
        Number m_ind_width; ///< The width of each individual subinterval.
    };
}
