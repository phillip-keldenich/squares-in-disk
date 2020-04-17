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

#include <doctest/doctest_fixed.hpp>
#include "ivarp/tuple.hpp"
#include "ivarp/number.hpp"

using namespace ivarp;

/// Compile time tests (correct types etc.)
struct NoCopy {
    explicit NoCopy(int i) noexcept : i{i} {}
    NoCopy(const NoCopy&) = delete;
    NoCopy& operator=(const NoCopy&) = delete;
    NoCopy(NoCopy&&) = default;
    NoCopy& operator=(NoCopy&&) = default;

    int i;
};

struct NoMove {
    explicit NoMove(int i) noexcept : i{i} {}
    NoMove(NoMove&&) = delete;
    NoMove(const NoMove&) = delete;
    NoMove& operator=(const NoMove&) = delete;
    NoMove& operator=(NoMove&&) = delete;
    int i;
};

/// A single integer in a tuple.
using Tuple1i = Tuple<int>;

/// Check that the type behaves correctly.
static_assert(std::is_trivially_default_constructible<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_destructible<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_copy_constructible<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_copy_assignable<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_copyable<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_constructible<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_move_constructible<Tuple1i>::value, "Error!");
static_assert(std::is_trivially_move_assignable<Tuple1i>::value, "Error!");

/// Check for an MSVC regression.
using Tuple17if = Tuple<IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, bool, IDouble,
                        IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, IFloat, char>;

static_assert(std::is_trivially_copy_constructible<Tuple17if>::value, "Error!");
static_assert(std::is_trivially_copy_assignable<Tuple17if>::value, "Error!");

/// A tuple containing a type that is not copyable.
using Tuple1s = Tuple<NoCopy>;
/// Check that the type behaves correctly.
static_assert(!std::is_trivially_default_constructible<Tuple1s>::value, "Error!");
static_assert(!std::is_copy_constructible<Tuple1s>::value, "Error!");
static_assert(!std::is_copy_assignable<Tuple1s>::value, "Error!");
static_assert(std::is_move_assignable<Tuple1s>::value, "Error!");
static_assert(std::is_move_constructible<Tuple1s>::value, "Error!");

/// A tuple containing a type that is neither copyable nor movable.
using Tuple1t = Tuple<NoMove>;
static_assert(!std::is_trivially_default_constructible<Tuple1t>::value, "Error!");
static_assert(!std::is_copy_constructible<Tuple1t>::value, "Error!");
static_assert(!std::is_copy_assignable<Tuple1t>::value, "Error!");
static_assert(!std::is_move_assignable<Tuple1t>::value, "Error!");
static_assert(!std::is_move_constructible<Tuple1t>::value, "Error!");

struct TestTupleSize {
    IFloat f;
    int i;
    double d;
    bool b;
};
static_assert(sizeof(Tuple<bool, double, int, IFloat>) == sizeof(TestTupleSize), "Size is wrong!");

namespace {
    Tuple1i t1i0;
    const Tuple1i t1i1{1};
    const Tuple1i t1i2{t1i1};
    const Tuple1i t1i3{std::move(t1i1)};
}

TEST_CASE("[ivarp][tuple] Simple tuple test") {
    REQUIRE(get<0>(t1i0) == 0);
    REQUIRE(get<0>(t1i1) == 1);
    REQUIRE(get<0>(t1i2) == 1);
    REQUIRE(get<0>(t1i3) == 1);
}

TEST_CASE("[ivarp][tuple] Noncopyable tuple test") {
    Tuple1s t1s{NoCopy(0)};
    REQUIRE(get<0>(t1s).i == 0);
}

namespace {
template<int I> struct TI { int test() { return I; }};

TEST_CASE("[ivarp][tuple] Long tuple test") {
    using TT = Tuple<TI<0>, TI<1>, TI<2>, TI<3>,
                     TI<4>, TI<5>, TI<6>, TI<7>,
                     TI<8>, TI<9>, TI<10>,TI<11>,
                     TI<12>,TI<13>,TI<14>,TI<15>,
                     TI<16>,TI<17>,TI<18>,TI<19>,
                     TI<20>,TI<21>,TI<22>,TI<23>,
                     TI<24>,TI<25>,TI<26>,TI<27>,
                     TI<28>,TI<29>,TI<30>,TI<31>, TI<32>>;

    TT t1;
    TT t2{};

#define DO_REQUIRE_TEST(i) REQUIRE(get<(i)>(t1).test() == get<(i)>(t2).test()); REQUIRE(t1[IVARP_IND(i)].test() == (i)); REQUIRE(get<i>(t1).test() == i)
    DO_REQUIRE_TEST(0);
    DO_REQUIRE_TEST(1);
    DO_REQUIRE_TEST(2);
    DO_REQUIRE_TEST(3);
    DO_REQUIRE_TEST(4);
    DO_REQUIRE_TEST(5);
    DO_REQUIRE_TEST(6);
    DO_REQUIRE_TEST(7);
    DO_REQUIRE_TEST(8);
    DO_REQUIRE_TEST(9);
    DO_REQUIRE_TEST(10);
    DO_REQUIRE_TEST(11);
    DO_REQUIRE_TEST(12);
    DO_REQUIRE_TEST(13);
    DO_REQUIRE_TEST(14);
    DO_REQUIRE_TEST(15);
    DO_REQUIRE_TEST(16);
    DO_REQUIRE_TEST(17);
    DO_REQUIRE_TEST(18);
    DO_REQUIRE_TEST(19);
    DO_REQUIRE_TEST(20);
    DO_REQUIRE_TEST(21);
    DO_REQUIRE_TEST(22);
    DO_REQUIRE_TEST(23);
    DO_REQUIRE_TEST(24);
    DO_REQUIRE_TEST(25);
    DO_REQUIRE_TEST(26);
    DO_REQUIRE_TEST(27);
    DO_REQUIRE_TEST(28);
    DO_REQUIRE_TEST(29);
    DO_REQUIRE_TEST(30);
    DO_REQUIRE_TEST(31);
    DO_REQUIRE_TEST(32);
#undef DO_REQUIRE_TEST
}

}
