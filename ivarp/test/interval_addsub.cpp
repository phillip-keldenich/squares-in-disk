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
// Created by Phillip Keldenich on 12.10.19.
//

#include <doctest/doctest_fixed.hpp>
#include "ivarp/number.hpp"
#include "test_util.hpp"

using namespace ivarp;

TEST_CASE_TEMPLATE("[ivarp][number] Simple interval addition tests (float/double)", FloatType, float, double) {
	const FloatType mi = std::numeric_limits<FloatType>::min();
	const FloatType ma = std::numeric_limits<FloatType>::max();
	const FloatType mam = std::nextafter(ma, FloatType(0));
	const FloatType inf = std::numeric_limits<FloatType>::infinity();
	using IV = Interval<FloatType>;

	IV exact{0.5,0.5};
	IV ex_r = exact + exact;
	REQUIRE(ex_r.same(IV{1,1}));

	IV small_pos{0, mi};
	IV small_mix{-mi,mi};
	IV large{ma,ma};
	IV la_r = large + small_pos;
	REQUIRE(la_r.same(IV{ma,inf}));
	la_r = large + small_mix;
	REQUIRE(la_r.same(IV{mam, inf}));
}

TEST_CASE_TEMPLATE("[ivarp][number] Simple interval subtraction tests (float/double)", FloatType, float, double) {
	const FloatType mi = std::numeric_limits<FloatType>::min();
	const FloatType ma = std::numeric_limits<FloatType>::max();
	const FloatType mam = std::nextafter(ma, FloatType(0));
	const FloatType inf = std::numeric_limits<FloatType>::infinity();
	using IV = Interval<FloatType>;

	IV exact{1.5,1.5};
	IV exact2{.5,.5};
	IV ex_r = exact - exact2;
	REQUIRE(ex_r.same(IV{1,1}));

	IV small_pos{0, mi};
	IV small_mix{-mi,mi};
	IV large{ma,ma};
	IV la_r = large - small_pos;
	REQUIRE(la_r.same(IV{mam,ma}));
	la_r = large - small_mix;
	REQUIRE(la_r.same(IV{mam, inf}));
}

TEST_CASE_TEMPLATE("[ivarp][number] Random interval additions and subtractions (float/double vs rational)", Float, float, double) {
	constexpr unsigned NUM_RANDOM_INTERVAL_ADD_SUB_TESTS = 65536;
	using IV = Interval<Float>;
	using IR = IRational;

	const IV width_classes[] = {
			{0, 0.1f},
			{0, 0.5f},
			{0.1f,0.3f},
			{0.5f,2.5f},
			{0.5f,5.f},
			{5.f,20.f},
			{20.f, 50.f}
	};
	const IV range{-100.,100.};

	for(unsigned j = 0; j < NUM_RANDOM_INTERVAL_ADD_SUB_TESTS; ++j) {
	    IV i1 = random_interval(range, std::begin(width_classes), std::end(width_classes));
		IV i2 = random_interval(range, std::begin(width_classes), std::end(width_classes));
	    IR ri1 = convert_number<IRational>(i1);
	    IR ri2 = convert_number<IRational>(i2);
		Rational rp1 = random_point(ri1);
		Rational rp2 = random_point(ri2);

		Rational rpp = rp1 + rp2;
		Rational rpm = rp1 - rp2;

		IV r1p2 = i1 + i2;
		IV r2p1 = i2 + i1;
		IR rr1p2 = ri1 + ri2;
		IR rr2p1 = ri2 + ri1;

		REQUIRE(r1p2.same(r2p1));
		REQUIRE(rr1p2.same(rr2p1));
		REQUIRE(rpp >= lb(r1p2));
		REQUIRE(rpp >= lb(rr1p2));
		REQUIRE(rpp <= ub(r1p2));
		REQUIRE(rpp <= ub(rr1p2));

		IV r1m2 = i1 - i2;
		IV r2m1 = i2 - i1;
		IR rr1m2 = ri1 - ri2;
		IR rr2m1 = ri2 - ri1;

		REQUIRE(r1m2.same(-r2m1));
		REQUIRE(rr1m2.same(-rr2m1));
		REQUIRE(rpm >= lb(r1m2));
		REQUIRE(rpm <= ub(r1m2));
		REQUIRE(rpm >= lb(rr1m2));
		REQUIRE(rpm <= ub(rr1m2));
	}
}
