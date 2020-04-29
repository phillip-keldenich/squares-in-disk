#include "../test_util.hpp"
#include "../cuda/cuda_test_util.cuh"
#include "../cuda/cuda_test_fn.cuh"
#include "ivarp/run_prover.hpp"

namespace {
	using namespace ivarp;
	using namespace ivarp::args;

	const auto G = x0 * sqrt(1_Z-square(x0)) + asin(x0);
    const auto est = G(maximum(x1 - x0 - 1_Z, -1_Z)) - G(ensure_expr(-1_Z)) + 2_Z*square(x0) - x0*x1;

	TEST_CASE("[ivarp][issue#9][cuda] asin test") {
		IDouble widths[] = {
			{0.0, 0.001}, {0.001, 0.01}, {0.01, 0.1}, {0.1, 0.5}, {0.5, 0.99}
		};

		Array<IDouble, 2> r1{IDouble{-1.0, 1.0}, IDouble{-1.0, 1.0}};
		for(int i = 0; i < 20; ++i) {
			cuda_test_fn(asin(x0), 16384, r1, std::begin(widths), std::end(widths));
			cuda_test_fn(G, 16384, r1, std::begin(widths), std::end(widths));
			cuda_test_fn(est, 16384, r1, std::begin(widths), std::end(widths));
		}
	}
}

