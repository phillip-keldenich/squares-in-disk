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
// Created by Phillip Keldenich on 29.11.19.
//

#pragma once

namespace ivarp {
    namespace impl {
		// Generally, our numbers should NEVER be NaN (that's why there is the undefinedness handling in intervals).
		// However, infinity * 0 results in a NaN, but we want a 0 in that case.
		template<typename F>
		    IVARP_HD static inline F nan_to_zero(F v) noexcept
		{
			return BOOST_UNLIKELY(v != v) ? F(0) : v;
		}

		template<bool NoInfinity>
		IVARP_HD static inline Bounds<float> ia_do_mul(float lb1, float lb2, float ub1, float ub2) noexcept {
#ifdef __CUDA_ARCH__
		    float lbr = __fmul_rd(lb1, lb2);
		    float ubr = __fmul_ru(ub1, ub2);
			return { NoInfinity ? lbr : nan_to_zero(lbr), NoInfinity ? ubr : nan_to_zero(ubr) };
#else
			__m128 t = _mm_set_ps(lb1, opacify(-ub1), 0.f, 0.f);
			__m128 o = _mm_set_ps(lb2, opacify(ub2), 0.f, 0.f);
			__m128 r = _mm_mul_ps(t, o); // NOLINT
			alignas(__m128) float rf[4];
			_mm_store_ps(rf, r);
			return { NoInfinity ? rf[3] : nan_to_zero(rf[3]), NoInfinity? -rf[2] : nan_to_zero(-rf[2]) };
#endif
		}

		template<bool NoInfinity>
		IVARP_HD static inline Bounds<double> ia_do_mul(double lb1, double lb2, double ub1, double ub2) noexcept {
#ifdef __CUDA_ARCH__
		    double lbr = __dmul_rd(lb1, lb2);
		    double ubr = __dmul_ru(ub1, ub2);
			return { NoInfinity ? lbr : nan_to_zero(lbr), NoInfinity ? ubr : nan_to_zero(ubr) };
#else
			__m128d t = _mm_set_pd(lb1, opacify(-ub1));
			__m128d o = _mm_set_pd(lb2, opacify(ub2));
			__m128d r = _mm_mul_pd(t, o); // NOLINT
			alignas(__m128d) double rd[2];
			_mm_store_pd(rd, r);
			return { NoInfinity ? rd[1] : nan_to_zero(rd[1]), NoInfinity ? -rd[0] : nan_to_zero(-rd[0]) };
#endif
		}

		template<typename FloatType,
		         typename B1 = fixed_point_bounds::Unbounded,
		         typename B2 = fixed_point_bounds::Unbounded>
		IVARP_HD static inline Bounds<FloatType>
			ia_mul_check_signs(const Interval<FloatType>& f1, const Interval<FloatType>& f2) noexcept
        {
			using namespace fixed_point_bounds;
            constexpr bool finite = is_finite(B1::lb, B1::ub) && is_finite(B2::lb, B2::ub);

            // f1 positive
            if (nonnegative<B1>(f1)) {
                if (nonnegative<B2>(f2)) {
                    // both positive
                    return ia_do_mul<finite>(lb(f1), lb(f2), ub(f1), ub(f2));
                } else if (nonpositive<B2>(f2)) {
                    // f1 positive, f2 negative
                    return ia_do_mul<finite>(ub(f1), lb(f2), lb(f1), ub(f2));
                } else {
                    // f1 positive, f2 mixed
                    return ia_do_mul<finite>(ub(f1), lb(f2), ub(f1), ub(f2));
                }
            }

            // f1 negative
            if (nonpositive<B1>(f1)) {
                if (nonnegative<B2>(f2)) {
                    // f1 negative, f2 positive
                    return ia_do_mul<finite>(lb(f1), ub(f2), ub(f1), lb(f2));
                } else if (nonpositive<B2>(f2)) {
                    // both negative
                    return ia_do_mul<finite>(ub(f1), ub(f2), lb(f1), lb(f2));
                } else {
                    // f1 negative, f2 mixed
                    return ia_do_mul<finite>(lb(f1), ub(f2), lb(f1), lb(f2));
                }
            }

            // f1 mixed
            if (nonnegative<B2>(f2)) {
                // f2 positive
                return ia_do_mul<finite>(lb(f1), ub(f2), ub(f1), ub(f2));
            } else if (nonpositive<B2>(f2)) {
                // f2 negative
                return ia_do_mul<finite>(ub(f1), lb(f2), lb(f1), lb(f2));
            }

            // both mixed
            Bounds<FloatType> r1 = ia_do_mul<finite>(lb(f1), ub(f2), lb(f1), lb(f2));
            Bounds<FloatType> r2 = ia_do_mul<finite>(lb(f2), ub(f1), ub(f1), ub(f2));
            if (r2.lb <= r1.lb) {
                r1.lb = r2.lb;
            }
            if (r1.ub <= r2.ub) {
                r1.ub = r2.ub;
            }
            return r1;
        }
	}

    template<> inline Interval<float>& Interval<float>::operator*=(IFloat o_) noexcept {
        impl::Bounds<float> b = impl::ia_mul_check_signs(*this, o_);
        this->m_lb = b.lb;
        this->m_ub = b.ub;
        m_undefined |= o_.m_undefined;
        return *this;
    }

    template<> inline Interval<double>& Interval<double>::operator*=(IDouble o_) noexcept {
	    impl::Bounds<double> b = impl::ia_mul_check_signs(*this, o_);
	    m_lb = b.lb;
	    m_ub = b.ub;
        m_undefined |= o_.m_undefined;
        return *this;
    }

    template<typename NumberType, typename Bounds1, typename Bounds2, typename = void> struct BoundedMul {
        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline NumberType eval(const NumberType& x1, const NumberType& x2)
                noexcept(std::is_nothrow_copy_constructible<NumberType>::value)
            {
                return x1 * x2;
            }
        )
    };

    template<typename FloatType, typename Bounds1, typename Bounds2>
        struct BoundedMul<Interval<FloatType>, Bounds1, Bounds2,
                          MakeVoid<std::enable_if_t<std::is_floating_point<FloatType>::value>>>
    {
        using NumberType = Interval<FloatType>;

        static inline IVARP_HD NumberType eval(const NumberType& x1, const NumberType& x2) noexcept {
            impl::Bounds<FloatType> b = impl::ia_mul_check_signs<FloatType,Bounds1,Bounds2>(x1, x2);
            return Interval<FloatType>{b.lb, b.ub, x1.possibly_undefined() || x2.possibly_undefined()};
        }
    };
}
