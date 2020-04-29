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
// Created by Phillip Keldenich on 05.10.19.
//

#pragma once

namespace ivarp {
    /// A unary tag implementing fixed non-negative integral powers, i.e., x ** u for some constant unsigned u.
    template<unsigned P> struct MathFixedPowTag {
        template<typename Context, typename NumberType>
        IVARP_HD static inline EnableForCUDANT<NumberType,NumberType> eval(const NumberType& n) noexcept;
        template<typename Context, typename NumberType>
        IVARP_H static inline DisableForCUDANT<NumberType,NumberType> eval(const NumberType& n);

        struct EvalBounds {
            template<typename B1> struct Eval {
                static constexpr std::int64_t lb = fixed_point_bounds::fpow_lb(B1::lb, B1::ub, P);
                static constexpr std::int64_t ub = fixed_point_bounds::fpow_ub(B1::lb, B1::ub, P);
            };
        };
    };

    /// Function that returns a math_fn implementing a ** P.
    template<unsigned P, typename A> static inline
        std::enable_if_t<IsMathExpr<A>::value, MathUnary<MathFixedPowTag<P>, BareType<A>>>
            fixed_pow(A&& a)
    {
        return MathUnary<MathFixedPowTag<P>, BareType<A>>{a};
    }

    /// Special case P == 2.
    template<typename A> static inline auto square(A&& a) {
        return fixed_pow<2>(std::forward<A>(a));
    }

    /// Function used to evaluate fixed_pow math_fns on floating-point values.
    template<unsigned P, typename FloatType>
        IVARP_HD static inline std::enable_if_t<std::is_floating_point<BareType<FloatType>>::value, FloatType>
            fixed_pow(FloatType f)
    {
        if(P > 1024) {
            return IVARP_NOCUDA_USE_STD pow(f, static_cast<FloatType>(P));
        } else {
            unsigned p = P;
            FloatType x = f;
            FloatType y = (p % 2) ? x : FloatType(1);
            for(p /= 2; p > 0; p /= 2) {
                x *= x;
                if(p % 2) { y *= x; }
            }
            return y;
        }
    }

    namespace impl {
        /// Raise lb, ub to the pth power without touching the rounding mode; if p is even, lb and ub are non-negative.
        template<bool Even> static inline IVARP_HD
            IFloat builtin_interval_pow(float lb, float ub, bool und, unsigned p) noexcept
        {
#if defined(__CUDA_ARCH__)
            float lbx = lb;
            float ubx = ub;
            float lby = Even ? 1.f : lb;
            float uby = Even ? 1.f : ub;
            for(p /= 2; p > 0; p /= 2) {
                lbx = __fmul_rd(lbx, lbx);
                ubx = __fmul_ru(ubx, ubx);
                if(p % 2) {
                    lby = __fmul_rd(lby, lbx);
                    uby = __fmul_ru(uby, ubx);
                }
            }
            return IFloat{lby, uby, und};
#else
            // If lb is negative, we must keep lb(x) and lb(y) negative.
            // If lb is positive, we must keep lb(x) and lb(y) positive.
            // If ub is negative, we must keep ub(x) and ub(y) positive and negate at the end.
            // If ub is positive, we must keep ub(x) and ub(y) negative and negate at the end.
            // Vectors are set up as [0, 0, ub, lb]
            do_opacify(lb);
            ub = -ub;
            do_opacify(ub);

            const __m128 negate = _mm_set_ps(0.f, 0.f, std::copysign(0.f, ub), std::copysign(0.f, lb));
            __m128 x = _mm_set_ps(0.f, 0.f, ub, lb);
            __m128 nx = _mm_xor_ps(x, negate);
            __m128 y = _mm_set_ps(0.f, 0.f, Even ? -1.f : ub, Even ? 1.f : lb);
            for(p /= 2; p > 0; p /= 2) {
                x = _mm_mul_ps(x, nx); // NOLINT
                nx = _mm_xor_ps(x, negate);
                if(p % 2) {
                    y = _mm_mul_ps(y, nx); // NOLINT
                }
            }
            alignas(__m128) float res[4];
            _mm_store_ps(res, y);
            return IFloat{res[0], -opacify(res[1]), und};
#endif
        }

        /// Raise lb, ub to the pth power without touching the rounding mode; if p is even, lb and ub are non-negative.
        /// Completely analogous to the float version above.
        template<bool Even> static inline IVARP_HD
            IDouble builtin_interval_pow(double lb, double ub, bool und, unsigned p)
        {
#if defined(__CUDA_ARCH__)
            double lbx = lb;
            double ubx = ub;
            double lby = Even ? 1. : lb;
            double uby = Even ? 1. : ub;
            for(p /= 2; p > 0; p /= 2) {
                lbx = __dmul_rd(lbx, lbx);
                ubx = __dmul_ru(ubx, ubx);
                if(p % 2) {
                    lby = __dmul_rd(lby, lbx);
                    uby = __dmul_ru(uby, ubx);
                }
            }
            return IDouble{lby, uby, und};
#else
            // If lb is negative, we must keep lb(x) and lb(y) negative.
            // If lb is positive, we must keep lb(x) and lb(y) positive.
            // If ub is negative, we must keep ub(x) and ub(y) positive and negate at the end.
            // If ub is positive, we must keep ub(x) and ub(y) negative and negate at the end.
            // Vectors are set up as [0, 0, ub, lb]
            do_opacify(lb);
            ub = -ub;
            do_opacify(ub);

            const __m128d negate = _mm_set_pd(std::copysign(0., ub), std::copysign(0., lb));
            __m128d x = _mm_set_pd(ub, lb);
            __m128d nx = _mm_xor_pd(x, negate);
            __m128d y = _mm_set_pd(Even ? -1. : ub, Even ? 1. : lb);
            for(p /= 2; p > 0; p /= 2) {
                x = _mm_mul_pd(x, nx); // NOLINT
                nx = _mm_xor_pd(x, negate);
                if(p % 2) {
                    y = _mm_mul_pd(y, nx); // NOLINT
                }
            }
            alignas(__m128d) double res[2];
            _mm_store_pd(res, y);
            return IDouble{res[0], -opacify(res[1]), und};
#endif
        }

        // rational integer powers
        extern IVARP_EXPORTED_SYMBOL IVARP_H Rational  rational_ipow(const Rational& r,  unsigned p);
        extern IVARP_EXPORTED_SYMBOL IVARP_H IRational rational_ipow(const IRational& r, unsigned p);
    }

    template<unsigned P, typename FloatType> static inline IVARP_HD
        std::enable_if_t<
            std::is_same<FloatType, float>::value || std::is_same<FloatType, double>::value,
            Interval<FloatType>
        > fixed_pow(Interval<FloatType> f) noexcept
    {
        if(P == 0) {
            return Interval<FloatType>{1,1,f.possibly_undefined()};
        } else if(P == 1u) {
            return f;
        } else if(P == 2u) {
            Interval<FloatType> r(f);
            r *= f;
            r.bound_from_below(0);
            return r;
        } else if(P % 2) {
            return impl::builtin_interval_pow<false>(f.lb(), f.ub(), f.possibly_undefined(), P);
        } else {
            if(ub(f) < 0) {
                return impl::builtin_interval_pow<true>(-f.ub(), -f.lb(), f.possibly_undefined(), P);
            } else if(lb(f) < 0) {
                FloatType maxabs = (ivarp::max)((ivarp::abs)(lb(f)), (ivarp::abs)(ub(f)));
                return impl::builtin_interval_pow<true>(FloatType(0), maxabs, f.possibly_undefined(), P);
            } else {
                return impl::builtin_interval_pow<true>(f.lb(), f.ub(), f.possibly_undefined(), P);
            }
        }
    }

    template<unsigned P, typename RType> static inline IVARP_H
        std::enable_if_t<
            std::is_same<RType, Rational>::value || std::is_same<RType, IRational>::value,
            RType
        > fixed_pow(const RType& r)
    {
        return impl::rational_ipow(r, P);
    }

    template<unsigned P> template<typename Context, typename NumberType>
        inline EnableForCUDANT<NumberType,NumberType> MathFixedPowTag<P>::eval(const NumberType& n) noexcept
    {
        return fixed_pow<P>(n);
    }

    template<unsigned P> template<typename Context, typename NumberType>
        inline DisableForCUDANT<NumberType,NumberType> MathFixedPowTag<P>::eval(const NumberType& n)
    {
        return fixed_pow<P>(n);
    }
}
