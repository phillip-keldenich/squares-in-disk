#pragma once

namespace ivarp {
namespace impl {
	/**
	 * For float, we are at most 4 ULP off the correctly rounded-to-nearest result.
	 */
	IVARP_D static inline float cuda_asin_rd(float x) noexcept {
	    float lb = (x >= 0) ? 0.f : -1.570796489715576171875f;
		x = asinf(x);
		for(int i = 0; i < 9; ++i) {
            x = rd_prev_float(x);
        }
		if(x < lb) {
			x = lb;
		}
		return x;
	}
	IVARP_D static inline float cuda_asin_ru(float x) noexcept {
        float ub = (x <= 0) ? 0.f : 1.570796489715576171875f;
        x = asinf(x);
        for(int i = 0; i < 9; ++i) {
            x = rd_next_float(x);
        }
        if(x > ub) {
            x = ub;
        }
        return x;
	}

	/**
	 * For double, we are at most 2 ULP off the correctly rounded-to-nearest result.
	 */
	IVARP_D static inline double cuda_asin_rd(double x) noexcept {
	    double lb = (x >= 0) ? 0. : -1.5707963267948967800435866593034006655216217041015625;
        x = asin(x);
        for(int i = 0; i < 5; ++i) {
            x = rd_prev_float(x);
        }
        if(x < lb) {
            x = lb;
        }
        return x;
	}
	IVARP_D static inline double cuda_asin_ru(double x) noexcept {
        double ub = (x <= 0) ? 0. :  1.5707963267948967800435866593034006655216217041015625;
        x = asin(x);
        for(int i = 0; i < 5; ++i) {
            x = rd_next_float(x);
        }
        if(x > ub) {
            x = ub;
        }
        return x;
	}
}
}

