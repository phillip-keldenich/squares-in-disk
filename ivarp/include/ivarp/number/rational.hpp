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
    IVARP_H inline BigInt to_bigint(unsigned long long l);
    IVARP_H inline BigInt to_bigint(long long l);

    template<> class Interval<Rational> {
    public:
        // we need to explicitly store infinities outside of the numbers.
        static constexpr bool has_explicit_infinity = true;

        // the number type (for external reference)
        using NumberType = Rational;

		IVARP_H explicit Interval() noexcept :
		    m_lb(0), m_ub(0), m_undefined(false),
		    m_lb_inf(false), m_ub_inf(false)
        {}

        template<typename T, std::enable_if_t<!std::is_same<BareType<T>, Interval>::value, int> = 0>
            IVARP_H explicit Interval(T&& num_type) :
                m_lb(num_type),
                m_ub(std::forward<T>(num_type)),
                m_undefined(false),
                m_lb_inf(false),
                m_ub_inf(false)
        {}

        Interval(const Interval&) = default;
        Interval &operator=(const Interval&) = default;

        IVARP_H Interval(Interval&& i) noexcept :
            m_lb(std::move(i.m_lb)), m_ub(std::move(i.m_ub)),
            m_undefined(i.m_undefined), m_lb_inf(i.m_lb_inf),
            m_ub_inf(i.m_ub_inf)
        {}

        IVARP_H Interval &operator=(Interval&& i) noexcept {
            m_lb = std::move(i.m_lb);
            m_ub = std::move(i.m_ub);
            m_undefined = i.m_undefined;
            m_lb_inf = i.m_lb_inf;
            m_ub_inf = i.m_ub_inf;
            return *this;
        }

        template<typename LBT, typename UBT>
            IVARP_H Interval(LBT&& lbt, UBT&& ubt, bool undef = false) :
                m_lb(to_number(std::forward<LBT>(lbt))),
                m_ub(to_number(std::forward<UBT>(ubt))), m_undefined(undef),
                m_lb_inf(std::is_same<BareType<LBT>,InfinityType>::value),
                m_ub_inf(std::is_same<BareType<UBT>,InfinityType>::value)
        {}

        template<typename R1, typename R2>
            IVARP_H Interval(R1&& lb, R2&& ub, bool undef, bool lbi, bool ubi) :
                m_lb(std::forward<R1>(lb)),
                m_ub(std::forward<R2>(ub)),
                m_undefined(undef), m_lb_inf(lbi), m_ub_inf(ubi)
        {}

        IVARP_H bool is_finite() const noexcept {
            return !m_lb_inf && !m_ub_inf;
        }

		IVARP_H bool finite_lb() const noexcept {
			return !m_lb_inf;
		}

		IVARP_H bool finite_ub() const noexcept {
			return !m_ub_inf;
		}

        IVARP_H Interval operator-() const {
            return {-this->m_ub, -this->m_lb, this->m_undefined, m_ub_inf, m_lb_inf };
        }

#define IVARP_DECLARE_OP(op)\
        IVARP_H inline Interval &operator op(const Interval& r);\
        template<typename T> IVARP_H std::enable_if_t<!std::is_same<BareType<T>, Interval>::value, Interval&>\
            operator op(const T& other)\
        {\
            return (*this op convert_number<Interval>(other));\
        }

        IVARP_DECLARE_OP(+=)
        IVARP_DECLARE_OP(-=)
        IVARP_DECLARE_OP(*=)
        IVARP_DECLARE_OP(/=)

#undef IVARP_DECLARE_OP

        IVARP_H const NumberType& lb() const noexcept { return this->m_lb; }
        IVARP_H const NumberType& ub() const noexcept { return this->m_ub; }
        IVARP_H NumberType& lb_ref() noexcept { return this->m_lb; }
        IVARP_H NumberType& ub_ref() noexcept { return this->m_ub; }

        /// Setting upper/lower bound to infinity/-infinity.
        IVARP_H void set_ub(const InfinityType&) noexcept {
            this->m_ub_inf = true;
            if(this->m_ub < this->m_lb) {
                this->m_ub = this->m_lb;
            }
        }
        IVARP_H void set_lb(const InfinityType&) noexcept {
            this->m_lb_inf = true;
            if(this->m_ub < this->m_lb) {
                this->m_lb = this->m_ub;
            }
        }

        /// Setting lb/ub to a value convertible to rational that does not have infinities.
        template<typename T>
            IVARP_H std::enable_if_t<!std::is_same<BareType<T>, InfinityType>::value &&
                                     !std::is_floating_point<BareType<T>>::value>
            set_lb(T&& u)
        {
            this->m_lb_inf = false;
            this->m_lb = std::forward<T>(u);
        }
        template<typename T>
            IVARP_H std::enable_if_t<!std::is_same<BareType<T>, InfinityType>::value &&
                                     !std::is_floating_point<BareType<T>>::value>
            set_ub(T&& u)
        {
            this->m_ub_inf = false;
            this->m_ub = std::forward<T>(u);
        }

        /// Settng lb/ub to a floating-point value, which may be infinite.
        template<typename T>
            IVARP_H std::enable_if_t<std::is_floating_point<BareType<T>>::value>
                set_lb(T u)
        {
            this->m_lb_inf = !std::isfinite(u);
            this->m_lb = u;
        }
        template<typename T>
            IVARP_H std::enable_if_t<std::is_floating_point<BareType<T>>::value>
                set_ub(T u)
        {
            this->m_ub_inf = !std::isfinite(u);
            this->m_ub = u;
        }

        IVARP_H void do_join(const Interval& other) {
            if(other.m_lb < this->m_lb) {
                this->m_lb = other.m_lb;
            }
            if(other.m_ub > this->m_ub) {
                this->m_ub = other.m_ub;
            }
            m_lb_inf |= other.m_lb_inf;
            m_ub_inf |= other.m_ub_inf;
            m_undefined |= other.m_undefined;
        }

        IVARP_H void do_join_defined(const Interval& other) {
            if(other.m_lb < this->m_lb) {
                this->m_lb = other.m_lb;
            }
            if(other.m_ub > this->m_ub) {
                this->m_ub = other.m_ub;
            }
            m_lb_inf |= other.m_lb_inf;
            m_ub_inf |= other.m_ub_inf;
        }

        IVARP_H bool same(const Interval& o) const noexcept {
            return same_lb(o) && same_ub(o) && m_undefined == o.m_undefined;
        }

        template<typename NumType>
            IVARP_H std::enable_if_t<!std::is_floating_point<BareType<NumType>>::value, bool>
                contains(const NumType& r) const noexcept
        {
            return (m_lb_inf || this->m_lb <= r) && (m_ub_inf || this->m_ub >= r);
        }

        template<typename NumType>
            IVARP_H std::enable_if_t<std::is_floating_point<BareType<NumType>>::value, bool>
                contains(const NumType& r) const noexcept
        {
            if(BOOST_UNLIKELY(std::isinf(r))) {
                return r > 0 ? m_ub_inf : m_lb_inf;
            } else {
                return (m_lb_inf || this->m_lb <= r) && (m_ub_inf || this->m_ub >= r);
            }
        }

        template<typename ON> static bool compare_to_rational(const ON& lhs, const Rational& rhs) {
            return lhs < rhs;
        }

        template<typename ON> static bool compare_to_rational(const Rational& lhs, const ON& rhs) {
            return lhs < rhs;
        }

        static bool compare_to_rational(long long lhs, const Rational& rhs) {
#if LONG_MAX == LLONG_MAX
            return long(lhs) < rhs;
#else
            return to_bigint(lhs) < rhs;
#endif
        }

        static bool compare_to_rational(const Rational& lhs, long long rhs) {
#if LONG_MAX == LLONG_MAX
            return lhs < long(rhs);
#else
            return lhs < to_bigint(rhs);
#endif
        }

        template<typename OtherNumType>
            IVARP_H bool below_lb(const OtherNumType& o) const noexcept
        {
            return !m_lb_inf && compare_to_rational(o, this->m_lb);
        }

        template<typename OtherNumType>
            IVARP_H bool below_ub(const OtherNumType& o) const noexcept
        {
            return m_ub_inf || compare_to_rational(o, this->m_ub);
        }

        template<typename OtherNumType>
            IVARP_H bool above_lb(const OtherNumType& o) const noexcept
        {
            return m_lb_inf || compare_to_rational(this->m_lb, o);
        }

        template<typename OtherNumType>
            IVARP_H bool above_ub(const OtherNumType& o) const noexcept
        {
            return !m_ub_inf && compare_to_rational(this->m_ub, o);
        }

        IVARP_H bool singleton() const noexcept {
            return !m_lb_inf && !m_ub_inf && this->m_lb == this->m_ub;
        }

        IVARP_H void set_bounds(const Interval& from) noexcept {
            this->m_lb = from.m_lb;
            this->m_ub = from.m_ub;
            this->m_lb_inf = from.m_lb_inf;
            this->m_ub_inf = from.m_ub_inf;
        }

        IVARP_HD void set_undefined(bool u) noexcept {
            m_undefined = u;
        }

        IVARP_HD bool possibly_undefined() const noexcept {
            return m_undefined;
        }

        IVARP_H ApplyBoundResult restrict_lower_bound(const Interval& other) {
            if(!other.finite_lb()) {
                return {false,false};
            }
            if(!this->finite_lb() || this->m_lb < other.m_lb) {
                if(this->finite_ub() && this->m_ub < other.m_lb) {
                    return {true, true};
                }
                this->m_lb = other.m_lb;
                this->m_lb_inf = false;
                return {true, false};
            }
            return {false,false};
        }

        IVARP_H ApplyBoundResult restrict_upper_bound(const Interval& other) {
            if(!other.finite_ub()) {
                return {false, false};
            }
            if(!this->finite_ub() || this->m_ub > other.m_ub) {
                if(this->finite_lb() && this->m_lb > other.m_ub) {
                    return {true, true};
                }
                this->m_ub = other.m_ub;
                this->m_ub_inf = false;
                return {true, false};
            }
            return {false, false};
        }

        IVARP_H static Interval combine_bounds(const Interval& lb_interval, const Interval& ub_interval) {
            Interval result;
            if(lb_interval.finite_lb()) {
                result.set_lb(lb_interval.lb());
            } else {
                result.set_lb(-infinity);
            }
            if(ub_interval.finite_ub()) {
                result.set_ub(ub_interval.ub());
            } else {
                result.set_ub(infinity);
            }
            return result;
        }

        IVARP_H bool empty() const noexcept {
            return finite_lb() && finite_ub() && this->m_ub < this->m_lb;
        }

        IVARP_H void set_bounds(const std::pair<NumberType,NumberType>& bs) noexcept {
            m_lb = bs.first;
            m_ub = bs.second;
        }

        /* this is easier to maintain if there is only one copy of the nearly identical code for each operator */
        #define IVARP_DEFINE_OP(op)\
			IVARP_H Interval operator op(const Interval& o) const {\
				Interval result{*this};\
				result op##= o;\
				return result;\
			}

        IVARP_DEFINE_OP(+)
        IVARP_DEFINE_OP(-)
        IVARP_DEFINE_OP(*)
        IVARP_DEFINE_OP(/)

		/* remove the macro again */
		#undef IVARP_DEFINE_OP

    private:
        IVARP_H bool same_lb(const Interval& other) const noexcept {
            return (m_lb_inf && other.m_lb_inf) || (!m_lb_inf && !other.m_lb_inf && this->m_lb == other.m_lb);
        }

        IVARP_H bool same_ub(const Interval& other) const noexcept {
            return (m_ub_inf && other.m_ub_inf) || (!m_ub_inf && !other.m_ub_inf && this->m_ub == other.m_ub);
        }

        template<typename T> static
            IVARP_H std::enable_if_t<!std::is_same<BareType<T>, InfinityType>::value, Rational>
                to_number(T&& t)
        {
            return std::forward<T>(t);
        }

        IVARP_H static Rational to_number(InfinityType) {
            return 0;
        }

        Rational m_lb, m_ub;
        bool m_undefined, m_lb_inf, m_ub_inf;
    };

    template<typename T> struct IsLongLong : std::false_type {};
    template<> struct IsLongLong<long long> : std::true_type {};
    template<> struct IsLongLong<unsigned long long> : std::true_type {};

    template<typename T> using DirectConversionToGMP =
        std::integral_constant<bool, !IsLongLong<T>::value && IsIntegral<T>::value>;

    template<typename IntType>
        static std::enable_if_t<DirectConversionToGMP<BareType<IntType>>::value, BigInt> to_bigint(const IntType& i)
    {
        return BigInt(i);
    }

    inline BigInt to_bigint(unsigned long long l) {
        mpz_t v;
        mpz_init(v);
        mpz_import(v, 1, 1, sizeof(unsigned long long), 0, 0, static_cast<const void*>(&l));
        BigInt result(v);
        mpz_clear(v);
        return result;
    }

    inline BigInt to_bigint(long long l) {
        mpz_t v;
        mpz_init(v);
        if(l < 0) {
            auto u = static_cast<unsigned long long>(-l);
            mpz_import(v, 1, 1, sizeof(unsigned long long), 0, 0, static_cast<const void*>(&u));
            mpz_neg(v, v);
        } else {
            mpz_import(v, 1, 1, sizeof(long long), 0, 0, static_cast<const void*>(&l));
        }
        BigInt result(v);
        mpz_clear(v);
        return result;
    }

    template<typename IntegralType1, typename IntegralType2>
        static inline std::enable_if_t<IsIntegral<IntegralType1>::value && IsIntegral<IntegralType2>::value, Rational>
            rational(IntegralType1 num, IntegralType2 denom)
    {
        Rational r{to_bigint(num), to_bigint(denom)};
        r.canonicalize();
        return r;
    }

    template<typename IntegralType1, std::enable_if_t<IsIntegral<IntegralType1>::value,int>>
        static inline Rational rational(IntegralType1 num)
    {
        return Rational{to_bigint(num)};
    }

    static inline const BigInt& opacify(const BigInt& b) noexcept { return b; }
    static inline const Rational &opacify(const Rational &r) noexcept { return r; }
    extern IVARP_EXPORTED_SYMBOL void invert(Rational& x) noexcept;
    extern IVARP_EXPORTED_SYMBOL Rational reciprocal(const Rational& x);

    template<typename T> struct ModfResult {
        using Type = T;
    };

    template<> struct ModfResult<Rational> {
        using Type = BigInt;
    };

    template<typename T> using ModfResultType = typename ModfResult<BareType<T>>::Type;

    template<typename T>
        std::enable_if_t<std::is_floating_point<BareType<T>>::value, T> modf(T x, ModfResultType<T>* intpart)
    {
        return std::modf(x, intpart);
    }

    extern IVARP_EXPORTED_SYMBOL Rational modf(const Rational& r, BigInt* intpart);

    inline IRational &Interval<Rational>::operator+=(const IRational& r) {
        this->m_lb += r.m_lb;
        this->m_ub += r.m_ub;
        this->m_lb_inf |= r.m_lb_inf;
        this->m_ub_inf |= r.m_ub_inf;
        this->m_undefined |= r.m_undefined;
        return *this;
    }

    inline IRational &Interval<Rational>::operator-=(const IRational& r) {
        this->m_lb -= r.m_ub;
        this->m_ub -= r.m_lb;
        this->m_lb_inf |= r.m_ub_inf;
        this->m_ub_inf |= r.m_lb_inf;
        this->m_undefined |= r.m_undefined;
        return *this;
    }

    namespace impl {
        extern IVARP_EXPORTED_SYMBOL
        std::tuple<Rational, Rational, bool, bool> ia_rational_mul(const IRational &r1, const IRational &r2);
    }

    inline IRational &Interval<Rational>::operator*=(const IRational& r) {
        std::tie(m_lb, m_ub, m_lb_inf, m_ub_inf) = impl::ia_rational_mul(*this, r);
        m_undefined |= r.m_undefined;
        return *this;
    }

    namespace impl {
        extern IVARP_EXPORTED_SYMBOL void irational_div(IRational& num, const IRational& den);
    }

    static inline void signs(const Interval<Rational>& r, int& sgnlb, int& sgnub) noexcept {
        sgnlb = r.finite_lb() ? sgn(r.lb()) : -1;
        sgnub = r.finite_ub() ? sgn(r.ub()) : 1;
    }

    inline IRational &Interval<Rational>::operator/=(const IRational& r) {
        impl::irational_div(*this, r);
        return *this;
    }
}
