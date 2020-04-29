#pragma once

#include <cfloat>
#include <climits>
#include <limits>
#include "ivarp/rounding.hpp"
#include "ivarp/cuda.hpp"

namespace ivarp {
	/**
	 * These behave like std::nextafter for incrementing/decrementing floats.
	 * There are two exceptions:
	 * - They are a lot faster.
	 * - They only work if rounding is set to round_down.
	 * - They cannot escape from infinities (going up from -inf yields -inf,
	 *   going down from inf yields inf), which is fine for our purposes.
	 */
	static inline float IVARP_HD rd_next_float(float f) {
#ifndef __CUDA_ARCH__
		const float dmn = -std::numeric_limits<float>::denorm_min();
		f = -f;
		opacify(f);
		return -opacify(f + dmn);
#else
		return __fadd_ru(f, 1.401298464324817071e-45); // note that relevant CUDA chips do not have penalties on denormal addition/multiplication
#endif
	}

	static inline double IVARP_HD rd_next_float(double d) {
#ifndef __CUDA_ARCH__
		const double dmn = -std::numeric_limits<double>::denorm_min();
		d = -d;
		opacify(d);
		return -opacify(d + dmn);
#else
		return __dadd_ru(d, 4.940656458412465442e-324);
#endif
	}

	static inline float IVARP_HD rd_prev_float(float f) {
#ifndef __CUDA_ARCH__
		const float dmn = std::numeric_limits<float>::denorm_min();
		return opacify(opacify(f) - dmn);
#else
		return __fsub_rd(f, 1.401298464324817071e-45);
#endif
	}

	static inline double IVARP_HD rd_prev_float(double d) {
#ifndef __CUDA_ARCH__
		const double dmn = std::numeric_limits<double>::denorm_min();
		return opacify(opacify(d) - dmn);
#else
		return __dsub_rd(d, 4.940656458412465442e-324);
#endif
	}
}
