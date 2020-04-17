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
// Created by Phillip Keldenich on 02.12.19.
//

#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <stdexcept>
#include <exception>
#include <array>
#include "ivarp/cuda.hpp"
#include "ivarp/metaprogramming.hpp"

/**
 * @brief Every part of IVARP is in this namespace.
 */
namespace ivarp {
    /// A CUDA-device compatible replacement for std::array<T,N>.
    template<typename T, std::size_t N> class Array {
    public:
        /// All the types are defined as for std::array.
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using iterator = T*;
        using const_iterator = const T*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        /// All constructors and destructors are, as for std::array, implicitly declared.

        /// Unlike std::array, have a static constexpr member with the size.
        static constexpr std::size_t length = N;

        IVARP_SUPPRESS_HD
        template<typename Iterator> IVARP_HD void initialize_from(Iterator beg)
        {
            for(std::size_t i = 0; i < N; ++i) {
                elements[i] = *beg;
                ++beg;
            }
        }

        /// Access with bounds checking.
        IVARP_H reference at(size_type pos) {
            if(pos >= N) {
                std::ostringstream msg;
                msg << "Array<T, " << N << ">::at( " << pos << " )";
                throw std::out_of_range(msg.str());
            }
            return (*this)[pos];
        }

        IVARP_H const_reference at(size_type pos) const {
            if(pos >= N) {
                std::ostringstream msg;
                msg << "Array<T, " << N << ">::at( " << pos << " ) const";
                throw std::out_of_range(msg.str());
            }
            return (*this)[pos];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr reference operator[](size_type pos) noexcept {
            static_assert(N > 0, "Accessing element in empty array!");
            return elements[pos];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr const_reference operator[](size_type pos) const noexcept {
            static_assert(N > 0, "Accessing element in empty array!");
            return elements[pos];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr reference front() noexcept {
            return elements[0];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr const_reference front() const noexcept {
            return elements[0];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr reference back() noexcept {
            return elements[N-1];
        }

        IVARP_SUPPRESS_HD
        IVARP_HD constexpr const_reference back() const noexcept {
            return elements[N-1];
        }

        IVARP_HD constexpr pointer data() noexcept {
            return &elements[0];
        }

        IVARP_HD constexpr const_pointer data() const noexcept {
            return &elements[0];
        }

        IVARP_HD constexpr iterator begin() noexcept {
            return data();
        }

        IVARP_HD constexpr const_iterator begin() const noexcept {
            return data();
        }

        IVARP_HD constexpr const_iterator cbegin() const noexcept {
            return data();
        }

        IVARP_HD constexpr iterator end() noexcept {
            return data() + N;
        }

        IVARP_HD constexpr const_iterator end() const noexcept {
            return data() + N;
        }

        IVARP_HD constexpr const_iterator cend() const noexcept {
            return data() + N;
        }

        IVARP_H constexpr reverse_iterator rbegin() noexcept {
            return reverse_iterator{data() + N - 1};
        }

        IVARP_H constexpr const_reverse_iterator rbegin() const noexcept {
            return const_reverse_iterator{data() + N - 1};
        }

        IVARP_H constexpr const_reverse_iterator crbegin() const noexcept {
            return rbegin();
        }

        IVARP_H constexpr reverse_iterator rend() noexcept {
            return reverse_iterator{data() - 1};
        }

        IVARP_H constexpr const_reverse_iterator rend() const noexcept {
            return const_reverse_iterator{data() - 1};
        }

        IVARP_H constexpr const_iterator crend() const noexcept {
            return rend();
        }

        IVARP_HD constexpr bool empty() const noexcept {
            return N == 0;
        }

        IVARP_HD constexpr size_type size() const noexcept {
            return N;
        }

        IVARP_HD constexpr size_type max_size() const noexcept {
            return N;
        }

        IVARP_SUPPRESS_HD
        IVARP_HD void fill(const T& value) noexcept(std::is_nothrow_copy_assignable<T>::value) {
            for(std::size_t i = 0; i < N; ++i) {
                elements[i] = value;
            }
        }

    private:
        template<typename TT> static IVARP_HD constexpr bool swap_is_noexcept(TT&) noexcept {
            using std::swap;
            return noexcept(swap(std::declval<TT&>(), std::declval<TT&>()));
        }

    public:
        IVARP_SUPPRESS_HD
        IVARP_HD void swap(Array& other) noexcept(swap_is_noexcept(std::declval<T&>())) {
            using std::swap;
            for(std::size_t i = 0; i < N; ++i) {
                swap(elements[i], other.elements[i]);
            }
        }

        value_type elements[N <= 0 ? 1 : N];
    };

    template<typename T> class Array<T,0u> {
    public:
        /// All the types are defined as for std::array.
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using iterator = T*;
        using const_iterator = const T*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        /// All constructors and destructors are, as for std::array, implicitly declared.

        /// Unlike std::array, have a static constexpr member with the size.
        static constexpr std::size_t length = 0;

        /// Access with bounds checking.
        IVARP_H reference at(size_type pos) {
            std::ostringstream msg;
            msg << "Array<T, 0>::at( " << pos << " )";
            throw std::out_of_range(msg.str());
        }

        IVARP_H const_reference at(size_type pos) const {
            std::ostringstream msg;
            msg << "Array<T, 0>::at( " << pos << " ) const";
            throw std::out_of_range(msg.str());
        }

        IVARP_HD constexpr pointer data() noexcept {
            return nullptr;
        }

        IVARP_HD constexpr const_pointer data() const noexcept {
            return nullptr;
        }

        IVARP_HD constexpr iterator begin() noexcept {
            return data();
        }

        IVARP_HD constexpr const_iterator begin() const noexcept {
            return data();
        }

        IVARP_HD constexpr const_iterator cbegin() const noexcept {
            return data();
        }

        IVARP_HD constexpr iterator end() noexcept {
            return nullptr;
        }

        IVARP_HD constexpr const_iterator end() const noexcept {
            return nullptr;
        }

        IVARP_HD constexpr const_iterator cend() const noexcept {
            return nullptr;
        }

        IVARP_H constexpr reverse_iterator rbegin() noexcept {
            return reverse_iterator{static_cast<pointer>(nullptr)};
        }

        IVARP_H constexpr const_reverse_iterator rbegin() const noexcept {
            return const_reverse_iterator{static_cast<pointer>(nullptr)};
        }

        IVARP_H constexpr const_reverse_iterator crbegin() const noexcept {
            return rbegin();
        }

        IVARP_H constexpr reverse_iterator rend() noexcept {
            return reverse_iterator{static_cast<pointer>(nullptr)};
        }

        IVARP_H constexpr const_reverse_iterator rend() const noexcept {
            return const_reverse_iterator{static_cast<pointer>(nullptr)};
        }

        IVARP_HD constexpr const_iterator crend() const noexcept {
            return rend();
        }

        IVARP_HD constexpr bool empty() const noexcept {
            return true;
        }

        IVARP_HD constexpr size_type size() const noexcept {
            return 0;
        }

        IVARP_HD constexpr size_type max_size() const noexcept {
            return 0;
        }

        IVARP_HD void fill(const T&) noexcept {}
        IVARP_HD void swap(Array&) noexcept {}
    };

    static_assert(std::is_trivial<Array<int,1>>::value, "Array<int> should be trivial!");

    template<std::size_t P, typename AT, std::size_t N>
        static inline IVARP_HD typename Array<AT,N>::reference get(Array<AT,N>& a) noexcept
    {
        static_assert(P < N, "Invalid Array access!");
        return a[P];
    }

    template<std::size_t P, typename AT, std::size_t N>
        static inline IVARP_HD typename Array<AT,N>::const_reference get(const Array<AT,N>& a) noexcept
    {
        static_assert(P < N, "Invalid Array access!");
        return a[P];
    }

    template<std::size_t P, typename AT, std::size_t N>
        static inline IVARP_HD AT&& get(Array<AT,N>&& a) noexcept
    {
        static_assert(P < N, "Invalid Array access!");
        return static_cast<AT&&>(a[P]);
    }

    template<std::size_t P, typename AT, std::size_t N>
        static inline IVARP_HD const AT&& get(const Array<AT,N>&& a) noexcept
    {
        static_assert(P < N, "Invalid Array access!");
        return static_cast<const AT&&>(a[P]);
    }

    template<typename ElementType, typename... Args>
        static inline Array<ElementType, sizeof...(Args)> make_array(Args&&... args)
    {
        return Array<ElementType, sizeof...(Args)>{ivarp::forward<Args>(args)...};
    }
}
