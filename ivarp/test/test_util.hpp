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
// Created by Phillip Keldenich on 11.10.19.
//

#pragma once

#include <doctest/doctest_fixed.hpp>
#include <random>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

#include "ivarp/number.hpp"

#define REQUIRE_SAME(x,y) do { const auto _vx = (x); const auto _vy = (y); if(!::ivarp::require_same(_vx, _vy)) { REQUIRE((x).same(y)); } } while(0)

namespace ivarp {
    template<typename I>
        static inline bool require_same(I i1, I i2)
    {
        if(!i1.same(i2)) {
            std::cout << "Unmet require_same: " << i1 << " is not the same as " << i2 << std::endl;
            return false;
        }
        return true;
    }

    extern std::mt19937_64 rng;

    template<typename NumberType, typename WidthClassRandomAccessIterator>
        static inline Interval<NumberType> random_interval(const Interval<NumberType>& range,
                                                           WidthClassRandomAccessIterator width_classes_begin,
                                                           WidthClassRandomAccessIterator width_classes_end)
    {
        const std::size_t n = std::distance(width_classes_begin, width_classes_end);
        std::uniform_int_distribution<std::size_t> class_selection(std::size_t(0), n-1u);
        std::size_t c = class_selection(rng);
        NumberType center = std::uniform_real_distribution<NumberType>(lb(range), ub(range))(rng);
        NumberType width = std::uniform_real_distribution<NumberType>(lb(width_classes_begin[c]), ub(width_classes_begin[c]))(rng);
        NumberType u = std::min(NumberType(center + 0.5f * width), ub(range));
        NumberType l = std::max(NumberType(center - 0.5f * width), lb(range));
        return Interval<NumberType>(l,u);
    }

    template<typename NumberType, typename WidthClassRandomAccessIterator>
        static inline Interval<NumberType> random_interval_norestrict(const Interval<NumberType>& range,
                                                           WidthClassRandomAccessIterator width_classes_begin,
                                                           WidthClassRandomAccessIterator width_classes_end)
    {
        const std::size_t n = std::distance(width_classes_begin, width_classes_end);
        std::uniform_int_distribution<std::size_t> class_selection(std::size_t(0), n-1u);
        std::size_t c = class_selection(rng);
        NumberType center = std::uniform_real_distribution<NumberType>(lb(range), ub(range))(rng);
        NumberType width = std::uniform_real_distribution<NumberType>(lb(width_classes_begin[c]), ub(width_classes_begin[c]))(rng);
        NumberType u = NumberType(center + 0.5f * width);
        NumberType l = NumberType(center - 0.5f * width);
        return Interval<NumberType>(l,u);
    }

    template<typename NumberType>
        static inline std::enable_if_t<!std::is_same<std::decay_t<NumberType>, Rational>::value, NumberType>
            random_point(const Interval<NumberType>& i)
    {
        std::uniform_real_distribution<NumberType> dist(lb(i), ub(i));
        return dist(rng);
    }

    static inline Rational random_point(const Interval<Rational>& r) {
        std::uniform_real_distribution<double> dist(r.lb().get_d(), r.ub().get_d());
        double d = dist(rng);
        Rational res(d);
        if(res < lb(r)) {
            return lb(r);
        } else if(res > ub(r)) {
            return ub(r);
        } else {
            return res;
        }
    }

    template<typename ConvertToType, typename FromType> ConvertToType tested_conversion(const FromType& f) {
        ConvertToType result = ivarp::convert_number<ConvertToType>(f);
        REQUIRE(lb(result) <= lb(f));
        REQUIRE(ub(result) >= ub(f));
        return result;
    }

    template<typename BinaryOperatorType, typename NumberType>
        static inline void random_test_binary_operator(const BinaryOperatorType& op,
                                                       const Interval<NumberType>& i1,
                                                       const Interval<NumberType>& i2, unsigned num_points)
    {
        Interval<NumberType> iop12 = op(i1, i2);
        REQUIRE(lb(iop12) <= ub(iop12));

        std::vector<Rational> points1, points2;
        for(unsigned i = 0; i < num_points; ++i) {
            points1.push_back(tested_conversion<Rational>(random_point(i1)));
            REQUIRE(i1.contains(points1.back()));
            points2.push_back(tested_conversion<Rational>(random_point(i2)));
            REQUIRE(i2.contains(points2.back()));
        }

        for(unsigned index1 = 0; index1 < num_points; ++index1) {
            for(unsigned index2 = 0; index2 < num_points; ++index2) {
                const Rational& a1 = points1[index1];
                const Rational& a2 = points2[index2];
                Rational rop12 = op(a1, a2);
                REQUIRE(iop12.contains(rop12));

                Rational a1c = a1 * 3 / 11;
                Rational a2c = a2 * 5 / 7;
                Interval<NumberType> ia1c = tested_conversion<Interval<NumberType>>(a1c);
                Interval<NumberType> ia2c = tested_conversion<Interval<NumberType>>(a2c);
                Rational rop12c = op(a1c, a2c);
                Interval<NumberType> op12c = op(ia1c, ia2c);
                REQUIRE(op12c.contains(rop12c));
            }
        }
    }

    template<typename NumberType, typename BinaryOperatorType, typename RangeClassIterator, typename WidthClassIterator>
        static inline void random_test_binary_operator(const BinaryOperatorType& op,
                                         RangeClassIterator rc_begin, RangeClassIterator rc_end,
                                         WidthClassIterator wc_begin, WidthClassIterator wc_end,
                                         unsigned num_random_intervals = 256, unsigned points_per_interval = 2)
    {
        using IV = Interval<NumberType>;

        std::uniform_int_distribution<std::size_t> rc_dist(0, std::distance(rc_begin, rc_end)-1u);
        std::vector<IV> ranges_n;

        for(unsigned j = 0; j < num_random_intervals; ++j) {
            std::size_t rc = rc_dist(rng);
            IDouble r = random_interval<double>(rc_begin[rc], wc_begin, wc_end);
            ranges_n.push_back(tested_conversion<IV>(r));
        }

        for(std::size_t r1 = 0; r1 < num_random_intervals; ++r1) {
            for(std::size_t r2 = 0; r2 < num_random_intervals; ++r2) {
                random_test_binary_operator(op, ranges_n[r1], ranges_n[r2], points_per_interval);
            }
        }
    }

    template<typename NumberType, typename BinaryOperatorType,
             typename Range1ClassIterator, typename Range2ClassIterator, typename WidthClassIterator>
        static inline void random_test_binary_operator(const BinaryOperatorType& op,
                                                   Range1ClassIterator rc1_begin, Range1ClassIterator rc1_end,
                                                   Range2ClassIterator rc2_begin, Range1ClassIterator rc2_end,
                                                   WidthClassIterator wc_begin, WidthClassIterator wc_end,
                                                   unsigned num_random_intervals = 256, unsigned points_per_interval = 2)
    {
        using IV = Interval<NumberType>;
        std::uniform_int_distribution<std::size_t> rc1_dist(0, std::distance(rc1_begin, rc1_end)-1u);
        std::uniform_int_distribution<std::size_t> rc2_dist(0, std::distance(rc2_begin, rc2_end)-1u);

        std::vector<IV> ranges1, ranges2;
        for(unsigned j = 0; j < num_random_intervals; ++j) {
            std::size_t rc1 = rc1_dist(rng);
            std::size_t rc2 = rc2_dist(rng);
            IDouble r1 = random_interval<double>(rc1_begin[rc1], wc_begin, wc_end);
            ranges1.push_back(tested_conversion<IV>(r1));
            IDouble r2 = random_interval<double>(rc2_begin[rc2], wc_begin, wc_end);
            ranges2.push_back(tested_conversion<IV>(r2));
        }

        for(std::size_t r1 = 0; r1 < num_random_intervals; ++r1) {
            for(std::size_t r2 = 0; r2 < num_random_intervals; ++r2) {
                random_test_binary_operator(op, ranges1[r1], ranges2[r2], points_per_interval);
            }
        }
    }

    template<typename NumberType, typename UnaryOperatorType, typename WidthClassIterator>
        static inline void random_test_unary_operator(
            const UnaryOperatorType& op,
            const IDouble& range,
            WidthClassIterator wc_begin,
            WidthClassIterator wc_end, unsigned num_tests = 65536)
    {
        using IV = Interval<NumberType>;
        using IR = IRational;
        using RP = Rational;

        for(unsigned j = 0; j < num_tests; ++j) {
            IV x = tested_conversion<IV>(random_interval(range, wc_begin, wc_end));
            IR xr = tested_conversion<IR>(x);
            REQUIRE(xr.lb() >= x.lb());
            REQUIRE(xr.ub() <= x.ub());

            RP rp = random_point(xr);
            REQUIRE(xr.contains(rp));

            // test with a tiny interval to catch rounding problems
            RP rpc = rp * 3 / 11;
            IV y = tested_conversion<IV>(rpc);
            IV op_x = op(x);
            IV op_y = op(y);
            RP op_rpc = op(rpc);
            REQUIRE(op_y.contains(op_rpc));

            IR ir_op_x = op(xr);
            RP rp_op_x = op(rp);
            REQUIRE(op_x.contains(rp_op_x));
            REQUIRE(ir_op_x.contains(rp_op_x));
            REQUIRE(ir_op_x.lb() >= op_x.lb());
            REQUIRE(ir_op_x.ub() <= op_x.ub());
        }
    }

    static inline std::int64_t random_int(std::int64_t begin, std::int64_t end) {
        std::uniform_int_distribution<std::int64_t> gen(begin, end-1);
        return gen(rng);
    }

    template<typename Printable> void require_printable_same(const Printable& p, const char* expected) {
        std::stringstream s;
        REQUIRE((s << p).good());
        REQUIRE(s.str() == std::string(expected));
    }

    template<class T> std::string type_name() {
        typedef typename std::remove_reference<T>::type TR;
        std::unique_ptr<char, void(*)(void*)> own(
    #ifndef _MSC_VER
                    abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
    #else
                    nullptr,
    #endif
                    std::free
        );
        std::string r = own != nullptr ? own.get() : typeid(TR).name();
        if (std::is_const<TR>::value)
            r += " const";
        if (std::is_volatile<TR>::value)
            r += " volatile";
        if (std::is_lvalue_reference<T>::value)
            r += "&";
        else if (std::is_rvalue_reference<T>::value)
            r += "&&";
        return r;
    }
}
