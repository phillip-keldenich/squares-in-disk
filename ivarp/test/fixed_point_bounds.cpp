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
// Created by Phillip Keldenich on 19.12.19.
//

#include "ivarp/number.hpp"
#include "test_util.hpp"
#include <doctest/doctest_fixed.hpp>

using namespace ivarp;

TEST_CASE("[ivarp][number] Fixed-point bounds, multiplication") {
    static_assert(fixed_point_bounds::fp_mul_rd(1'250000, 3'000000) == 3'750000, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_mul_ru(1'250000, 3'000000) == 3'750000, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_mul_rd(1'250000, 3'001000) == 3'751250, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_mul_ru(1'250000, 3'001000) == 3'751250, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_mul_rd(1'250000, 3'000010) == 3'750012, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_mul_ru(1'250000, 3'000010) == 3'750013, "Incorrect result!");

    std::int64_t buffer[32];
    for(int i = 0; i < 16; ++i) {
        // no overflow with these numbers
        buffer[i] = random_int(-INT64_C(3'000'000'000), INT64_C(3'000'000'000));
    }
    for(int i = 16; i < 32; ++i) {
        // may cause overflows
        buffer[i] = random_int(-INT64_C(300'000'000'000'000), INT64_C(300'000'000'000'000));
    }

    Rational product_den = rational(1, fixed_point_bounds::denom() * fixed_point_bounds::denom());
    for(int i = 0; i < 32; ++i) {
        for(int j = 0; j < 32; ++j) {
            std::int64_t fp_rd = fixed_point_bounds::fp_mul_rd(buffer[i], buffer[j]);
            std::int64_t fp_ru = fixed_point_bounds::fp_mul_ru(buffer[i], buffer[j]);

            REQUIRE(fp_rd <= fp_ru);
            if(i < 16 && j < 16) {
                REQUIRE(fixed_point_bounds::is_lb(fp_rd));
                REQUIRE(fixed_point_bounds::is_ub(fp_ru));
            }

            Rational lb_rat = rational(fp_rd, fixed_point_bounds::denom());
            Rational ub_rat = rational(fp_ru, fixed_point_bounds::denom());
            Rational product = rational(buffer[i]) * rational(buffer[j]) * product_den;

            if(fixed_point_bounds::is_lb(fp_rd)) {
                REQUIRE(product >= lb_rat);
            }
            if(fixed_point_bounds::is_ub(fp_ru)) {
                REQUIRE(product <= ub_rat);
            }
        }
    }
}

TEST_CASE("[ivarp][number] Fixed-point bounds, division") {
    static_assert(fixed_point_bounds::fp_div_rd(fixed_point_bounds::fp_maxint() * fixed_point_bounds::denom(), 1) ==
                                                fixed_point_bounds::max_bound(), "Incorrect result!");
    static_assert(fixed_point_bounds::fp_div_ru(fixed_point_bounds::fp_maxint() * fixed_point_bounds::denom(), 1) ==
                                                fixed_point_bounds::max_bound(), "Incorrect result!");
    static_assert(fixed_point_bounds::fp_div_rd(1'250000, 250000) == 5'000000, "Incorrect result!");
    static_assert(fixed_point_bounds::fp_div_ru(1'250000, 250000) == 5'000000, "Incorrect result!");

    /// Check the case where numerator * 1000 would overflow and we actually need the 128-bit/64-bit long div.
    static_assert(fixed_point_bounds::fp_div_rd(4611686018427'387904, 20'571000) ==  224183851948'246944,
                  "Incorrect result!");
    static_assert(fixed_point_bounds::fp_div_ru(4611686018427'387904, 20'571000) ==  224183851948'246945,
                  "Incorrect result!");

    std::int64_t buffer[32];
    for(int i = 0; i < 8; ++i) {
        // overflow may occur with these numbers as denom
        do {
            buffer[i] = random_int(-INT64_C(1'000), INT64_C(1'000));
        } while(buffer[i] == 0);
    }
    for(int i = 8; i < 16; i += 2) {
        // overflow should not occur
        buffer[i] = random_int(1'000, INT64_C(5000000000000000));
        buffer[i+1] = -buffer[i];
    }
    for(int i = 16; i < 32; ++i) {
        do {
            buffer[i] = random_int(-INT64_C(300'000'000'000'000), INT64_C(300'000'000'000'000));
        } while(buffer[i] == 0);
    }

    for(int i = 0; i < 32; ++i) {
        for(int j = 0; j < 32; ++j) {
            std::int64_t fp_rd = fixed_point_bounds::fp_div_rd(buffer[i], buffer[j]);
            std::int64_t fp_ru = fixed_point_bounds::fp_div_ru(buffer[i], buffer[j]);

            if(j >= 8) {
                REQUIRE(fixed_point_bounds::is_lb(fp_rd));
                REQUIRE(fixed_point_bounds::is_ub(fp_ru));
            }

            Rational lb_rat = rational(fp_rd, fixed_point_bounds::denom());
            Rational ub_rat = rational(fp_ru, fixed_point_bounds::denom());
            Rational quot = rational(buffer[i]) / rational(buffer[j]);

            if(fixed_point_bounds::is_lb(fp_rd)) {
                REQUIRE(quot >= lb_rat);
            }
            if(fixed_point_bounds::is_ub(fp_ru)) {
                REQUIRE(quot <= ub_rat);
            }
        }
    }
}

TEST_CASE("[ivarp][number] Fixed-point interval order") {
    using namespace fixed_point_bounds;

    static_assert(iv_order(-2, -1, 0, 1)  == Order::LT, "Incorrect order!");
    static_assert(iv_order(-2, -1, -1, 1) == Order::LE, "Incorrect order!");
    static_assert(iv_order(min_bound(), -1, 0, 1) == Order::LT, "Incorrect order!");
    static_assert(iv_order(min_bound(), -1, 0, max_bound()) == Order::LT, "Incorrect order!");
    static_assert(iv_order(min_bound(), 0, 0, max_bound()) == Order::LE, "Incorrect order!");
    static_assert(iv_order(min_bound(), max_bound(), min_bound(), max_bound()) == Order::UNKNOWN, "Incorrect order!");
    static_assert(iv_order(-100, -100, -100, -100) == Order::EQ, "Incorrect result!");
    static_assert(iv_order(0, 3, 1, 2) == Order::UNKNOWN, "Incorrect result!");
    static_assert(iv_order(0, 2, 1, 3) == Order::UNKNOWN, "Incorrect result!");
    static_assert(iv_order(0, 1, 2, 3) == Order::LT, "Incorrect result!");
    static_assert(iv_order(1, 3, 0, 2) == Order::UNKNOWN, "Incorrect result!");
    static_assert(iv_order(1, 2, 0, 3) == Order::UNKNOWN, "Incorrect result!");
    static_assert(iv_order(2, 3, 0, 1) == Order::GT, "Incorrect result!");

    std::int64_t buffer[16];
    for (unsigned i = 0; i < 16; ++i) {
        buffer[i] = random_int(-INT64_C(300'000'000), INT64_C(300'000'000));
    }
    for (unsigned bi : buffer) {
        for (unsigned bj : buffer) {
            for (std::int64_t bi2 : buffer) {
                if (bi2 < bi) {
                    continue;
                }

                for (std::int64_t bj2 : buffer) {
                    if (bj2 < bj) {
                        continue;
                    }

                    Order ord = iv_order(bi, bi2, bj, bj2);
                    double di = bi;
                    double dj = bj;
                    double di2 = bi2;
                    double dj2 = bj2;

                    IDouble idi(di, di2), idj(dj, dj2);
                    if (ord == Order::EQ) {
                        REQUIRE(definitely(idi == idj));
                    }
                    else if (ord == Order::LE) {
                        REQUIRE(definitely(idi <= idj));
                        REQUIRE(!definitely(idi < idj));
                    }
                    else if (ord == Order::LT) {
                        REQUIRE(definitely(idi < idj));
                    }
                    else if (ord == Order::GE) {
                        REQUIRE(definitely(idi >= idj));
                        REQUIRE(!definitely(idi > idj));
                    }
                    else if (ord == Order::GT) {
                        REQUIRE(definitely(idi > idj));
                    }
                    else {
                        REQUIRE(possibly(idi < idj));
                        REQUIRE(possibly(idi > idj));
                    }
                }
            }
        }
    }
}

TEST_CASE("[ivarp][number] Fixed point bounds, sqrt") {
    using namespace fixed_point_bounds;
    REQUIRE(fp_iv_sqrt(-1, 1).lb == min_bound());
    REQUIRE(fp_iv_sqrt(-1, 1).ub == max_bound());
    REQUIRE(fp_sqrt_rd(0) == 0);
    REQUIRE(fp_sqrt_ru(0) == 0);
    REQUIRE(fp_sqrt_rd(int_to_fp(1)) == int_to_fp(1));
    REQUIRE(fp_sqrt_ru(int_to_fp(1)) == int_to_fp(1));
    REQUIRE(fp_sqrt_rd(int_to_fp(2)) == 1'414213);
    REQUIRE(fp_sqrt_ru(int_to_fp(2)) == 1'414214);
    REQUIRE(fp_sqrt_rd(int_to_fp(3)) == 1'732050);
    REQUIRE(fp_sqrt_ru(int_to_fp(3)) == 1'732051);
    REQUIRE(fp_sqrt_rd(int_to_fp(4)) == int_to_fp(2));
    REQUIRE(fp_sqrt_ru(int_to_fp(4)) == int_to_fp(2));
    REQUIRE(fp_sqrt_rd(int_to_fp(5)) == 2'236067);
    REQUIRE(fp_sqrt_ru(int_to_fp(5)) == 2'236068);
    REQUIRE(fp_sqrt_rd(int_to_fp(6)) == 2'449489);
    REQUIRE(fp_sqrt_ru(int_to_fp(6)) == 2'449490);
    REQUIRE(fp_sqrt_rd(int_to_fp(7)) == 2'645751);
    REQUIRE(fp_sqrt_ru(int_to_fp(7)) == 2'645752);
    REQUIRE(fp_sqrt_rd(int_to_fp(8)) == 2'828427);
    REQUIRE(fp_sqrt_ru(int_to_fp(8)) == 2'828428);
    REQUIRE(fp_sqrt_rd(int_to_fp(9)) == int_to_fp(3));
    REQUIRE(fp_sqrt_ru(int_to_fp(9)) == int_to_fp(3));
}

TEST_CASE("[ivarp][number] Fixed point bounds, fixed_pow") {
    using namespace fixed_point_bounds;

    REQUIRE(fpow_lb(int_to_fp(-5), int_to_fp(4), 2) == 0);
    REQUIRE(fpow_ub(int_to_fp(-5), int_to_fp(4), 2) == int_to_fp(25));
    REQUIRE(fpow_lb(int_to_fp(-100), int_to_fp(2), 3) == int_to_fp(-1000000));
    REQUIRE(fpow_ub(int_to_fp(-100), int_to_fp(0), 3) == 0);
    REQUIRE(fpow_ub(int_to_fp(-100), int_to_fp(2), 3) == int_to_fp(8));
    REQUIRE(fpow_lb(int_to_fp(-10), int_to_fp(-6), 4) == int_to_fp(1296));
    REQUIRE(fpow_ub(int_to_fp(-10), int_to_fp(-6), 4) == int_to_fp(10000));
}

TEST_CASE("[ivarp][number] Fixed point bounds, fixed_pow vs. sqrt") {
    using namespace fixed_point_bounds;

    for(unsigned i = 0; i < 32; ++i) {
        std::int64_t x = random_int(INT64_C(0), INT64_C(300'000'000));
        std::int64_t sqrt_lb = fp_sqrt_rd(x);
        std::int64_t sqrt_ub = fp_sqrt_ru(x);
        std::int64_t pow_lb = fpow_lb(sqrt_lb, sqrt_ub, 2);
        std::int64_t pow_lb2 = fpow_lb(-sqrt_ub, -sqrt_lb, 2);
        REQUIRE(pow_lb == pow_lb2);
        std::int64_t pow_ub = fpow_ub(sqrt_lb, sqrt_ub, 2);
        std::int64_t pow_ub2 = fpow_ub(-sqrt_ub, -sqrt_lb, 2);
        REQUIRE(pow_ub == pow_ub2);
        REQUIRE(pow_lb <= x);
        REQUIRE(pow_ub >= x);
    }
}
