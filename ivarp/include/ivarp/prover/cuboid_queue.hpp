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
// Created by Phillip Keldenich on 12.11.19.
//

#pragma once
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "ivarp/metaprogramming.hpp"

namespace ivarp {
    enum class CuboidQueueOrder {
        LIFO, FIFO
    };

    template<typename ElementType, bool Locking, CuboidQueueOrder Order> class CuboidQueue {
    public:
        using OrderTag = std::integral_constant<CuboidQueueOrder, Order>;
        using FifoTag = std::integral_constant<CuboidQueueOrder, CuboidQueueOrder::FIFO>;
        using LifoTag = std::integral_constant<CuboidQueueOrder, CuboidQueueOrder::LIFO>;
        using Container = std::conditional_t<Order==CuboidQueueOrder::FIFO,
                                             std::deque<ElementType>,std::vector<ElementType>>;

        /// Create a queue with init_size default-constructed entries on which num_threads threads work.
        explicit CuboidQueue(std::size_t init_size, std::size_t num_threads) :
            elements(init_size), num_threads(num_threads), idle_threads(0)
        {}

        /// Direct, unlocked access to the elements. Useful for the parallel initialization phase.
        ElementType& operator[](std::size_t i) noexcept {
            return elements[i];
        }
        const ElementType& operator[](std::size_t i) const noexcept {
            return elements[i];
        }

        std::mutex &mutex() const noexcept {
            return lock;
        }

        /// Take the next element off the queue and assign it to e.
        /// Returns false iff the queue is empty and all threads are done; in that case, e is left unchanged.
        bool pop_into(ElementType& e) {
            return do_take_into(e, std::integral_constant<bool, Locking>{}, OrderTag{});
        }

        /// Take the next (up to) num_elements out of the queue into output.
        /// Returns without changing the size of output if the queue is empty and all threads are done.
        void pop_into(std::vector<ElementType>& output, std::size_t num_elements) {
            do_take_into(output, std::integral_constant<bool, Locking>{}, OrderTag{}, num_elements);
        }

        /// Add a single element to the queue/stack.
        template<typename... Args> void emplace(Args&&... args) {
            LockGuard l(lock);
            elements.emplace_back(std::forward<Args>(args)...);
            if(Locking && idle_threads > 0) {
                condition.notify_one();
            }
        }

        /// Enqueue all elements from [b,e).
        template<typename Iterator> void enqueue_bulk(Iterator b, Iterator e)
        {
            LockGuard l(lock);
            for(; b != e; ++b) {
                elements.emplace_back(*b);
            }
            if(Locking && idle_threads > 0 && !elements.empty()) {
                condition.notify_all();
            }
        }

        /// Non-locking method to clear the entire queue.
        void clear() noexcept {
            return elements.clear();
        }

        /// Non-locking iterator methods.
        auto cbegin() const noexcept {
            return elements.cbegin();
        }
        auto begin() noexcept {
            return elements.begin();
        }
        auto cend() const noexcept {
            return elements.cend();
        }
        auto end() noexcept {
            return elements.end();
        }
        void unlocked_erase(typename Container::iterator start, typename Container::iterator end) {
            elements.erase(start, end);
        }

        /// A (locked if Locking is true) size check.
        std::size_t size() const noexcept {
            LockGuard l(lock);
            return elements.size();
        }

        /// A size check without external locking.
        std::size_t unlocked_size() const noexcept {
            return elements.size();
        }

        bool peek(ElementType& out) const noexcept {
            LockGuard l(lock);
            return do_peek(OrderTag{}, out);
        }

        bool unlocked_peek(ElementType& out) const noexcept {
            return do_peek(OrderTag{}, out);
        }

    private:
        struct NoOpLockGuard {
            explicit NoOpLockGuard(std::mutex&) noexcept {}
        };

        using LockGuard = std::conditional_t<Locking, std::unique_lock<std::mutex>, NoOpLockGuard>;

        bool do_peek(FifoTag, ElementType& out) const noexcept {
            if(elements.empty()) {
                return false;
            }
            out = elements.front();
            return true;
        }

        bool do_peek(LifoTag, ElementType& out) const noexcept {
            if(elements.empty()) {
                return false;
            }
            out = elements.back();
            return true;
        }

        template<typename TagType>
            bool do_take_into(ElementType& e, std::false_type /*lock*/, TagType t) noexcept
        {
            if(elements.empty()) {
                return false;
            } else {
                return do_take_into_nonempty(e, t);
            }
        }

        void nonempty_take_next(std::vector<ElementType>& out, FifoTag) {
            out.push_back(elements.front());
            elements.pop_front();
        }

        void nonempty_take_next(std::vector<ElementType>& out, LifoTag) {
            out.push_back(elements.back());
            elements.pop_back();
        }

        template<typename Ord>
        void do_take_into(std::vector<ElementType>& output, std::false_type /* lock */, Ord o,
                          std::size_t num_elements)
        {
            if (num_elements > elements.size()) {
                num_elements = elements.size();
            }
            for (std::size_t i = 0; i < num_elements; ++i) {
                nonempty_take_next(output, o);
            }
        }

        bool wait_if_empty(LockGuard& l) {
            if (elements.empty()) {
                // if we are the last thread that potentially could have enqueued new elements, we are done.
                // wake all other threads and return false.
                if (++idle_threads >= num_threads) {
                    condition.notify_all();
                    return false;
                }

                // otherwise, there may still be threads that may be going to insert new elements; wait.
                condition.wait(l, [&]() noexcept -> bool { return idle_threads >= num_threads || !elements.empty(); });
                if (elements.empty()) {
                    return false;
                } else {
                    --idle_threads;
                }
            }
            return true;
        }

        template<typename Ord>
        void do_take_into(std::vector<ElementType>& output, std::true_type /* lock */, Ord o,
                          std::size_t num_elements)
        {
            LockGuard l(lock);
            if (!wait_if_empty(l)) {
                return;
            }
            if (num_elements > elements.size()) {
                num_elements = elements.size();
            }
            for (std::size_t i = 0; i < num_elements; ++i) {
                nonempty_take_next(output, o);
            }
        }

        template<typename TagType>
            bool do_take_into(ElementType& e, std::true_type /*lock*/, TagType t) noexcept
        {
            LockGuard l(lock);
            if (!wait_if_empty(l)) {
                return false;
            }
            return do_take_into_nonempty(e, t);
        }

        template<typename = void> bool do_take_into_nonempty(ElementType& e, FifoTag /*lock*/) noexcept {
            e = std::move(elements.front());
            elements.pop_front();
            return true;
        }

        template<typename = void> bool do_take_into_nonempty(ElementType& e, LifoTag /*lock*/) noexcept {
            e = std::move(elements.back());
            elements.pop_back();
            return true;
        }

        Container elements;
        mutable std::mutex lock;
        std::condition_variable condition;
        std::size_t num_threads;
        std::size_t idle_threads;
    };
}
