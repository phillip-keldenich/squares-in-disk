#include "ivarp/math.hpp"
#include "ivarp_cuda/memory.hpp"
#include "ivarp_cuda/error.hpp"
#include "../test_util.hpp"
#include <limits>
#include <cfloat>

namespace {
	using namespace ivarp;
	using namespace ivarp::cuda;

	void __global__ kernel_test_next_float(int* result, float* problem) {
		if(*result) {
			return;
		}

		float begin, end;
		int sgn = blockIdx.x == 0 ? 1 : -1;
		int exp = threadIdx.x - 127;
		if(sgn > 0) {
			if(exp == -127) {
				begin = 0.f;
				end = ldexpf(1.f, -126);
			} else {
				begin = ldexpf(1.0f, exp);
				end = (exp == 127) ? HUGE_VALF : ldexpf(1.0f, exp+1);
			}
		} else {
			if(exp == -127) {
				begin = -ldexpf(1.f, -126);
				end = 0.f;
			} else {
				begin = (exp == 127) ? -FLT_MAX : -ldexpf(1.0f, exp);
				end = ldexpf(1.0f, exp-1);
			}
		}

		int cnt = 65536;
		for(float x = begin; x < end; ) {
			float next = rd_next_float(x);
			float next2 = nextafterf(x, HUGE_VALF);

			if(x >= next || next != next2) {
				if(atomicCAS(result, 0, 1) == 0) {
					*problem = x;
				}
				return;
			}

			if(--cnt == 0) {
				if(*result) {
					return;
				}
				// *cnt = 65536;
				return; // removing this would check all floats but that times out.
			}

			x = next;
		}
	}

	TEST_CASE("[ivarp][math][cuda] Check rd_next_float vs nextafterf") {
		DeviceArray<int> result(1);
		DeviceArray<float> problem(1);
		problem[0] = std::numeric_limits<float>::quiet_NaN();
		result[0] = 0;
		kernel_test_next_float<<<2,255>>>(result.pass_to_device(), problem.pass_to_device());
		throw_if_cuda_error("Synchronous launch error", cudaPeekAtLastError());
		throw_if_cuda_error("Asynchronous launch error", cudaDeviceSynchronize());
		result.read_from_device();
		problem.read_from_device();
		if(result[0] != 0) {
			std::cerr << "Problematic value: " << problem[0] << std::endl;
		}
		REQUIRE(result[0] == 0);
		REQUIRE((bool)(problem[0] != problem[0]));
	}
}

