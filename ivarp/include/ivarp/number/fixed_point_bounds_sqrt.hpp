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
// Created by Phillip Keldenich on 10.01.20.
//

#pragma once

namespace ivarp {
namespace fixed_point_bounds {
    struct FixedPointBounds {
        std::int64_t lb;
        std::int64_t ub;
    };

    constexpr IVARP_HD static inline std::int64_t fp_sqrt_bisect_rd(std::int64_t x, std::int64_t lb, std::int64_t ub) {
        // invariant: ru(ub * ub) >= x, ru(lb * lb) < x
        while(ub - lb > 1) {
            std::int64_t mid = (lb + ub) / 2;
            std::int64_t sqr = fp_mul_ru(mid,mid);
            if(sqr >= x) {
                ub = mid;
            } else {
                lb = mid;
            }
        }
        if(fp_mul_ru(ub,ub) == x) {
            return ub;
        }
        return lb;
    }

    constexpr IVARP_HD static inline std::int64_t fp_sqrt_bisect_ru(std::int64_t x, std::int64_t lb, std::int64_t ub) {
        // invariant: rd(ub * ub) > x, rd(lb * lb) <= x
        while(ub - lb > 1) {
            std::int64_t mid = (lb + ub) / 2;
            std::int64_t sqr = fp_mul_rd(mid,mid);
            if(sqr > x) {
                ub = mid;
            } else {
                lb = mid;
            }
        }
        if(fp_mul_rd(lb,lb) == x) {
            return lb;
        }
        return ub;
    }

    constexpr IVARP_HD static inline std::int64_t fp_sqrt_rd(std::int64_t x) noexcept {
        if(x == 0) {
            return 0;
        } else if(x >= max_bound()) {
            return 3037000'499976;
        } else if(x < denom()) {
            return fp_sqrt_bisect_rd(x, x, denom());
        } else if(x <= int_to_fp(16)) {
            return fp_sqrt_bisect_rd(x, x / 4, x);
        } else {
            // prevent overflows during bisection by using ru(x/4) as upper bound for large values
            return fp_sqrt_bisect_rd(x, 0, x / 4 + 1);
        }
    }

    constexpr IVARP_HD static inline std::int64_t fp_sqrt_ru(std::int64_t x) noexcept {
        if(x == 0) {
            return 0;
        } else if(x >= max_bound()) {
            return max_bound();
        } else if(x < denom()) {
            return fp_sqrt_bisect_ru(x, x, denom());
        } else if(x <= int_to_fp(16)) {
            return fp_sqrt_bisect_ru(x, x / 4, x);
        } else {
            return fp_sqrt_bisect_ru(x, 0, x / 4 + 1);
        }
    }

    constexpr IVARP_HD static inline FixedPointBounds fp_iv_sqrt(std::int64_t lb, std::int64_t ub) {
        if(lb < 0) {
            return FixedPointBounds{min_bound(),max_bound()};
        } else {
            return FixedPointBounds{fp_sqrt_rd(lb), fp_sqrt_ru(ub)};
        }
    }

    template<typename Bounds> struct SqrtEvalBounds {
    private:
        static constexpr auto bounds = fp_iv_sqrt(Bounds::lb, Bounds::ub);

    public:
        static constexpr std::int64_t lb = bounds.lb;
        static constexpr std::int64_t ub = bounds.ub;
    };
}
}

