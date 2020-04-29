#include "test_util.hpp"
#include "ivarp/math.hpp"

namespace {
using namespace ivarp;
/* // this takes quite a while but works
TEST_CASE("[ivarp][math] Walk through 32 bit floats") {
	for(float f = -std::numeric_limits<float>::max(); f < std::numeric_limits<float>::infinity();) {
		float stdtest = std::nextafterf(f, std::numeric_limits<float>::infinity());
		float next = ivarp::rd_next_float(f);
		REQUIRE(stdtest == next);
		f = next;
	}
	for(float f = std::numeric_limits<float>::max(); f > -std::numeric_limits<float>::infinity();) {
		float stdtest = std::nextafterf(f, std::numeric_limits<float>::infinity());
		float next = ivarp::rd_prev_float(f);
		REQUIRE(stdtest == next);
		f = next;
	}
}
*/

float stdincrement(float f) {
	return std::nextafterf(f, std::numeric_limits<float>::infinity());
}

float stddecrement(float f) {
	return std::nextafterf(f, -std::numeric_limits<float>::infinity());
}

double stdincrement(double d) {
	return std::nextafter(d, std::numeric_limits<double>::infinity());
}

double stddecrement(double d) {
	return std::nextafter(d, -std::numeric_limits<double>::infinity());
}

template<typename NT> void incdectest(NT x) {
	REQUIRE(stdincrement(x) == rd_next_float(x));
	REQUIRE(stddecrement(x) == rd_prev_float(x));
}

TEST_CASE_TEMPLATE("[ivarp][math] Next float/prev float tests", NT, float, double) {
	REQUIRE(rd_next_float(std::numeric_limits<NT>::infinity()) == std::numeric_limits<NT>::infinity());
	REQUIRE(rd_next_float(-std::numeric_limits<NT>::infinity()) == -std::numeric_limits<NT>::infinity());
	REQUIRE(rd_next_float(std::numeric_limits<NT>::max()) == std::numeric_limits<NT>::infinity());
	REQUIRE(rd_prev_float(-std::numeric_limits<NT>::max()) == -std::numeric_limits<NT>::infinity());
	REQUIRE(rd_prev_float(std::numeric_limits<NT>::infinity()) == std::numeric_limits<NT>::infinity());
	REQUIRE(rd_prev_float(-std::numeric_limits<NT>::infinity()) == -std::numeric_limits<NT>::infinity());
	REQUIRE(rd_next_float(NT(0)) == std::numeric_limits<NT>::denorm_min());
	REQUIRE(rd_prev_float(NT(0)) == -std::numeric_limits<NT>::denorm_min());
	incdectest(NT(-1));
	incdectest(NT(1));
	incdectest(NT(10000000));
	incdectest(-NT(10000000));
	incdectest(std::numeric_limits<NT>::min());
	incdectest(std::numeric_limits<NT>::max());
	incdectest(-std::numeric_limits<NT>::min());
	incdectest(-std::numeric_limits<NT>::max());
	incdectest(std::numeric_limits<NT>::denorm_min());
	incdectest(-std::numeric_limits<NT>::denorm_min());
}

}

