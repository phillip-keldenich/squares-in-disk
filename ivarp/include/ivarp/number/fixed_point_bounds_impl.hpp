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
// Created by Phillip Keldenich on 20.12.19.
//

#pragma once

namespace ivarp {
namespace fixed_point_bounds {
namespace fimpl {
    /// Note that this function is not for use at runtime; there are builtins/more efficient versions for that purpose.
    IVARP_HD static inline constexpr unsigned number_of_leading_zeros(std::uint64_t x) noexcept {
        unsigned n = 0;
        if(x == 0) return 64;

        if(x <= UINT64_C(0x0000'0000'FFFF'FFFF)) {
            n += 32;
            x <<= 32;
        }
        if(x <= UINT64_C(0x0000'FFFF'FFFF'FFFF)) {
            n += 16;
            x <<= 16;
        }
        if(x <= UINT64_C(0x00FF'FFFF'FFFF'FFFF)) {
            n += 8;
            x <<= 8;
        }
        if(x <= UINT64_C(0x0FFF'FFFF'FFFF'FFFF)) {
            n += 4;
            x <<= 4;
        }
        if (x <= UINT64_C(0x3FFF'FFFF'FFFF'FFFF)) {
            n += 2;
            x <<= 2;
        }
        if (x <= UINT64_C(0x7FFF'FFFF'FFFF'FFFF)) {
            n += 1;
        }
        return n;
    }

    IVARP_HD static inline constexpr std::int64_t fp_uadd(std::uint64_t a, std::uint64_t b) noexcept {
        return (a+b >= static_cast<std::uint64_t>(max_bound())) ? max_bound() : static_cast<std::int64_t>(a+b);
    }

    IVARP_HD static inline constexpr std::int64_t fp_add(std::int64_t a, std::int64_t b) noexcept {
        return (a < 0 && b < 0) ? -fp_uadd(-a, -b) :
               (a > 0 && b > 0) ? fp_uadd(a, b) :
               a + b;
    }

    IVARP_HD static inline constexpr std::int64_t saturated_umul(std::uint64_t a, std::uint64_t b) noexcept {
        return (b == 0) ? 0 :
               (a > static_cast<std::uint64_t>(max_bound())/b) ? max_bound() :
               static_cast<std::int64_t>(a*b);
    }

    IVARP_HD static inline constexpr std::int64_t fp_do_mul_rd(std::int64_t a, std::int64_t b) noexcept {
        std::int64_t ia = a / denom();
        std::int64_t ib = b / denom();
        std::int64_t fa = a % denom();
        std::int64_t fb = b % denom();
        std::int64_t ip = saturated_umul(saturated_umul(ia, ib), denom());
        std::int64_t f1p = fp_uadd(saturated_umul(ia,fb), saturated_umul(ib,fa));
        std::int64_t f2p = (fa * fb) / denom();
        return fp_uadd(fp_uadd(ip, f1p), f2p);
    }

    IVARP_HD static inline constexpr std::int64_t fp_do_mul_ru(std::int64_t a, std::int64_t b) noexcept {
        std::int64_t ia = a / denom();
        std::int64_t ib = b / denom();
        std::int64_t fa = a % denom();
        std::int64_t fb = b % denom();
        std::int64_t ip = saturated_umul(saturated_umul(ia, ib), denom());
        std::int64_t f1p = fp_uadd(saturated_umul(ia,fb), saturated_umul(ib,fa));
        std::int64_t f2p = (fa * fb) / denom() + ((fa * fb) % denom() != 0);
        return fp_uadd(fp_uadd(ip, f1p), f2p);
    }

    IVARP_HD static inline constexpr std::int64_t fp_do_long_div(std::uint64_t a, std::uint64_t b, std::uint64_t r) {
        // multiply m = a * denom; this requires extra bits
        std::uint32_t a1(a >> 32);
        std::uint32_t a2(a);
        std::uint64_t m2 = static_cast<std::uint64_t>(a2) * denom();
        std::uint64_t m1 = static_cast<std::uint64_t>(a1) * denom() + (m2 >> 32);

        // the high 32 and low 64 bits of the extended numerator
        std::uint64_t num_high = m1 >> 32;
        std::uint64_t num_low  = ((m1 & 0xffffffff)<<32) | (m2 & 0xffffffff);
        if(num_high >= b) {
            // overflow: result does not fit 64-bit unsigned
            return max_bound();
        }

        // the algorithm needs the divisor to start with a '1' bit;
        // adjust for that by shifting
        unsigned normalize_shift = number_of_leading_zeros(b);
        if(normalize_shift > 0) {
            b <<= normalize_shift;
            num_high <<= normalize_shift;
            num_high |= num_low >> (64u - normalize_shift);
            num_low <<= normalize_shift;
        }

        // based on Knuth's Algorithm 'D' long division
        std::uint64_t qhat = num_high / (b >> 32);
        std::uint64_t rhat = num_high % (b >> 32);
        while((qhat >> 32) != 0 || (qhat & ~UINT32_C(0)) * (b & ~UINT32_C(0)) > ((rhat << 32) | (num_low >> 32))) {
            qhat -= 1;
            rhat += std::uint32_t(b >> 32);
            if((rhat >> 32) != 0u) {
                break;
            }
        }
        std::uint32_t quot_high(qhat);

        std::uint64_t uhat = ((num_high << 32) | (num_low >> 32)) - quot_high * b;
        qhat = uhat / (b >> 32);
        rhat = uhat % (b >> 32);
        while((qhat >> 32) != 0 || (qhat & ~UINT32_C(0)) * (b & ~UINT32_C(0)) > ((rhat << 32) | (num_low >> 32))) {
            qhat -= 1;
            rhat += std::uint32_t(b >> 32);
            if((rhat >> 32) != 0u) {
                break;
            }
        }
        std::uint32_t quot_low(qhat);

        std::uint64_t quot = (std::uint64_t(quot_high) << 32) | quot_low;
        std::uint64_t rem = (((uhat << 32) | (num_low & ~UINT32_C(0))) - quot_low * b) >> normalize_shift;
        if(quot >= static_cast<std::uint64_t>(max_bound())) {
            return max_bound();
        }
        return quot + r * (rem != 0);
    }

    IVARP_HD static inline constexpr std::int64_t fp_do_div_rd(std::int64_t a, std::int64_t b) noexcept {
        return (b >= max_bound()) ? 0 : fp_do_long_div(a, b, 0);
    }

    IVARP_HD static inline constexpr std::int64_t fp_do_div_ru(std::int64_t a, std::int64_t b) noexcept {
        return (b == 0 || a >= max_bound()) ? max_bound() : fp_do_long_div(a, b, 1);
    }
}
}
}
