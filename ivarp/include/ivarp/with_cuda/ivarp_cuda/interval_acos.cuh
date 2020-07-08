#pragma once

namespace ivarp {
namespace impl {
	/**
	 * For float, we are at most 3 ULP off the correctly rounded-to-nearest result.
	 */
	IVARP_D static inline float cuda_acos_rd(float x) noexcept {
	    float lb = 0.0f;
		x = acosf(x);
		for(int i = 0; i < 7; ++i) {
            x = rd_prev_float(x);
        }
		if(x < lb) {
			x = lb;
		}
		return x;
	}
	IVARP_D static inline float cuda_acos_ru(float x) noexcept {
        float ub = 3.1415927410125732421875f;
        x = acosf(x);
        for(int i = 0; i < 7; ++i) {
            x = rd_next_float(x);
        }
        if(x > ub) {
            x = ub;
        }
        return x;
	}

	/***
	 * For double, we are at most 1 ULP off the correctly rounded-to-nearest result.
	 */
	IVARP_D static inline double cuda_acos_rd(double x) noexcept {
        x = acos(x);
        for(int i = 0; i < 3; ++i) {
            x = rd_prev_float(x);
        }
        if(x < 0.0) {
            x = 0.0;
        }
        return x;
	}
	IVARP_D static inline double cuda_acos_ru(double x) noexcept {
		double ub = 3.141592653589793560087173318606801331043243408203125;
        x = acos(x);
        for(int i = 0; i < 3; ++i) {
            x = rd_next_float(x);
        }
        if(x > ub) {
            x = ub;
        }
        return x;
	}
}
}

