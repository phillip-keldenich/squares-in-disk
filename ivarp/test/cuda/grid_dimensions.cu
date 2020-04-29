#include "../test_util.hpp"
#include "ivarp/run_prover.hpp"

namespace {
	using namespace ivarp;
	using namespace ivarp::impl;
	using namespace ivarp::impl::cuda;

	TEST_CASE("[ivarp][cuda] Grid size and dimensions - 2 static") {
		using Seq = SplitInfoSequence<SplitInfo<2, 128>, SplitInfo<3,128>>;
        REQUIRE(get_grid_y(Seq{}) == 1);
        dim3 r64 = pack_factors_2(64, 128, 128);
        REQUIRE(r64.z == 1);
        REQUIRE(r64.y == 8);
        REQUIRE(r64.x == 8);

	    dim3 r100 = pack_factors_2(100, 128, 128);
	    REQUIRE(r100.z == 1);
	    REQUIRE(r100.y == 10);
	    REQUIRE(r100.x == 10);
	}

	TEST_CASE("[ivarp][cuda] Grid size and dimensions - 3 static") {
        using Seq = SplitInfoSequence<SplitInfo<2, 128>, SplitInfo<3, 128>, SplitInfo<4, 64>>;
        REQUIRE(get_grid_y(Seq{}) == 1);
        dim3 r256 = pack_factors_3(256, 128, 128, 4);
        REQUIRE(r256.z == 8);
        REQUIRE(r256.y == 8);
        REQUIRE(r256.x == 4);
	}
}

