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
// Created by Phillip Keldenich on 10.10.19.
//

#pragma once

#include "ivarp/array.hpp"
#include "ivarp/rounding.hpp"
#include "device_compat.hpp"
#include "traits.hpp"

namespace ivarp {
    /// Is returned by various methods that restrict bounds.
    struct ApplyBoundResult {
        bool bound_changed;
        bool result_empty;

        IVARP_HD ApplyBoundResult& operator|=(ApplyBoundResult o) noexcept {
            bound_changed |= o.bound_changed;
            result_empty |= o.result_empty;
            return *this;
        }

        IVARP_HD ApplyBoundResult operator|(ApplyBoundResult o) const noexcept {
            return {bound_changed || o.bound_changed, result_empty || o.result_empty};
        }
    };

    /// An interval class.
    template<typename NumType> class Interval {
    public:
        static constexpr bool has_explicit_infinity = false;

        // check that we are not using the infinity-aware-NumType version for Rational intervals.
        static_assert(std::is_floating_point<NumType>::value, "Regular intervals use floating-point types!");

        // the number type (for external reference)
        using NumberType = NumType;

        explicit IVARP_HD Interval(NumberType n) noexcept :
            m_lb(n), m_ub(n), m_undefined(false)
        {}

        template<typename LBT, typename UBT>
            IVARP_HD Interval(LBT&& l, UBT&& u, bool undef = false) noexcept :
                m_lb(to_lb(std::forward<LBT>(l))), m_ub(to_ub(std::forward<UBT>(u))), m_undefined(undef)
        {}

        Interval() noexcept = default;
        Interval(const Interval& i) noexcept = default;
        Interval &operator=(const Interval& i) noexcept = default;
        Interval(Interval&& i) noexcept = default;
        Interval &operator=(Interval&& i) noexcept = default;

        IVARP_HD void do_join_defined(const Interval& other) noexcept {
            if(other.m_lb < this->m_lb) {
                this->m_lb = other.m_lb;
            }
            if(other.m_ub > this->m_ub) {
                this->m_ub = other.m_ub;
            }
        }

        IVARP_HD void do_join(const Interval& other) noexcept {
            if(other.m_lb < this->m_lb) {
                this->m_lb = other.m_lb;
            }
            if(other.m_ub > this->m_ub) {
                this->m_ub = other.m_ub;
            }
            this->m_undefined |= other.m_undefined;
        }

        IVARP_HD void bound_from_below(const NumType& n) noexcept {
            if(this->m_lb < n) {
                this->m_lb = n;
            }
        }

        IVARP_HD void bound_from_above(const NumType& n) noexcept {
            if(this->m_ub > n) {
                this->m_ub = n;
            }
        }

        /// Restrict this interval to be at most other.ub().
        /// Note that the interval is never made empty.
        IVARP_HD ApplyBoundResult restrict_upper_bound(const Interval& other) noexcept {
            if(other.m_ub < this->m_ub) {
                if(other.m_ub < this->m_lb) {
                    return {true, true};
                }
                this->m_ub = other.m_ub;
                return {true, false};
            }
            return {false,false};
        }

        /// Restrict this interval to be at least other.lb().
        /// Note that the interval is never made empty.
        IVARP_HD ApplyBoundResult restrict_lower_bound(const Interval& other) noexcept {
            if(other.m_lb > this->m_lb) {
                if(other.m_lb > this->m_ub) {
                    return {true, true};
                }
                this->m_lb = other.m_lb;
                return {true, false};
            }
            return {false,false};
        }

        template<typename OtherNumType>
            IVARP_HD EnableForCudaNT<OtherNumType, bool> below_lb(const OtherNumType& o) const noexcept
        {
            return exact_less_than(o, this->m_lb);
        }

        template<typename OtherNumType>
            IVARP_H DisableForCudaNT<OtherNumType, bool> below_lb(const OtherNumType& o) const
        {
            return exact_less_than(o, this->m_lb);
        }

        template<typename OtherNumType>
            IVARP_HD EnableForCudaNT<OtherNumType, bool> above_lb(const OtherNumType& o) const noexcept
        {
            return exact_less_than(this->m_lb, o);
        }

        template<typename OtherNumType>
            IVARP_H DisableForCudaNT<OtherNumType, bool> above_lb(const OtherNumType& o) const
        {
            return exact_less_than(this->m_lb, o);
        }

        template<typename OtherNumType>
            IVARP_HD EnableForCudaNT<OtherNumType, bool> above_ub(const OtherNumType& o) const noexcept
        {
            return exact_less_than(this->m_ub, o);
        }

        template<typename OtherNumType>
            IVARP_H DisableForCudaNT<OtherNumType, bool> above_ub(const OtherNumType& o) const
        {
            return exact_less_than(this->m_ub, o);
        }

        template<typename OtherNumType>
            IVARP_HD EnableForCudaNT<OtherNumType, bool> below_ub(const OtherNumType& o) const noexcept
        {
            return exact_less_than(o, this->m_ub);
        }

        template<typename OtherNumType>
            IVARP_H DisableForCudaNT<OtherNumType, bool> below_ub(const OtherNumType& o) const
        {
            return exact_less_than(o, this->m_ub);
        }

        IVARP_HD bool singleton() const noexcept {
            return this->m_lb == this->m_ub;
        }

        IVARP_HD bool is_finite() const noexcept {
            return IVARP_NOCUDA_USE_STD isfinite(this->m_lb) && IVARP_NOCUDA_USE_STD isfinite(this->m_ub);
        }

		IVARP_HD bool finite_lb() const noexcept {
			return IVARP_NOCUDA_USE_STD isfinite(this->m_lb);
		}

		IVARP_HD bool finite_ub() const noexcept {
			return IVARP_NOCUDA_USE_STD isfinite(this->m_ub);
		}

        IVARP_HD void set_ub(const InfinityType&) noexcept {
            this->m_ub = impl::inf_value<NumberType>();
        }

        IVARP_HD void set_lb(const InfinityType&) noexcept {
            this->m_lb = -impl::inf_value<NumberType>();
        }

        template<typename OtherNumType> IVARP_HD
	    	std::enable_if_t<!IsRational<OtherNumType>::value, bool> contains(const OtherNumType& n) const noexcept
	    {
            return !exact_less_than(n, this->m_lb) && !exact_less_than(this->m_ub, n);
        }

        template<typename OtherNumType> IVARP_HD
            std::enable_if_t<IsRational<OtherNumType>::value, bool> contains(const OtherNumType& n) const noexcept
        {
            // protect against passing -inf or inf into GMP (which causes a deliberate division-by-zero)
            return (this->m_lb == -impl::inf_value<NumberType>() || this->m_lb <= n) &&
                   (this->m_ub == impl::inf_value<NumberType>() || this->m_ub >= n);
        }

        IVARP_HD Interval operator-() const noexcept {
            return {-this->m_ub, -this->m_lb, this->m_undefined};
        }

        IVARP_HD bool same(const Interval& o) const noexcept {
            return this->m_lb == o.m_lb && this->m_ub == o.m_ub && this->m_undefined == o.m_undefined;
        }

        IVARP_HD void set_bounds(const Interval& from) noexcept {
            this->m_lb = from.m_lb;
            this->m_ub = from.m_ub;
        }

#define IVARP_DECLARE_OP(op)\
        IVARP_HD inline Interval &operator op(Interval o) noexcept;\
        template<typename T> IVARP_HD std::enable_if_t<!std::is_same<BareType<T>, Interval>::value, Interval&>\
            operator op(T&& other) noexcept {\
                return (*this op convert_number<Interval>(std::forward<T>(other)));\
            }

        IVARP_DECLARE_OP(+=)
        IVARP_DECLARE_OP(-=)
        IVARP_DECLARE_OP(*=)
        IVARP_DECLARE_OP(/=)
#undef IVARP_DECLARE_OP

        /// Combine the lower bound from lb_interval and the upper bound from ub_interval into a new interval,
        /// without checking/guaranteeing non-emptiness.
        static Interval combine_bounds(const Interval& lb_interval, const Interval& ub_interval) noexcept {
            return {lb_interval.lb(), ub_interval.ub()};
        }

        /// Normally, our intervals should _NOT_ be empty.
        bool empty() const noexcept {
            return this->m_ub < this->m_lb;
        }

        IVARP_HD bool possibly_undefined() const noexcept {
            return m_undefined;
        }

        IVARP_HD void set_undefined(bool und = true) noexcept {
            m_undefined = und;
        }

        IVARP_HD const NumType& lb() const noexcept {
            return m_lb;
        }

        IVARP_HD const NumType& ub() const noexcept {
            return m_ub;
        }

        IVARP_HD void set_lb(const NumberType& n) noexcept {
            m_lb = n;
        }

        IVARP_HD void set_ub(const NumberType& n) noexcept {
            m_ub = n;
        }

        IVARP_HD void set_bounds(const std::pair<NumberType,NumberType>& bs) noexcept {
            m_lb = bs.first;
            m_ub = bs.second;
        }

        IVARP_HD void set_bounds(const impl::Bounds<NumberType>& bs) noexcept {
            m_lb = bs.lb;
            m_ub = bs.ub;
        }

        IVARP_HD void set_bounds(std::pair<NumberType,NumberType>&& bs) noexcept {
            m_lb = std::move(bs.first);
            m_ub = std::move(bs.second);
        }

        IVARP_HD Interval join(const Interval& d) const noexcept {
            Interval result{*this};
            result.do_join(d);
            return result;
        }

        /* this is easier to maintain if there is only one copy of the nearly identical code for each operator */
        #define IVARP_DEFINE_OP(op)\
			IVARP_HD Interval operator op(const Interval& o) const noexcept {\
				Interval result{*this};\
				result op##= o;\
				return result;\
			}\
			template<typename T> IVARP_HD std::enable_if_t<!std::is_same<BareType<T>, Interval>::value, Interval>\
			    operator op(T&& o) noexcept\
			{\
			    return (*this op convert_number<Interval>(std::forward<T>(o)));\
			}

        IVARP_DEFINE_OP(+)
        IVARP_DEFINE_OP(-)
        IVARP_DEFINE_OP(*)
        IVARP_DEFINE_OP(/)

		/* remove the macro again */
		#undef IVARP_DEFINE_OP

    private:
        template<typename FromType> IVARP_HD static
            std::enable_if_t<!std::is_same<BareType<FromType>,InfinityType>::value, NumberType>
                to_lb(FromType&& f) noexcept
        {
            return std::forward<FromType>(f);
        }

        template<typename FromType> IVARP_HD static
            std::enable_if_t<!std::is_same<BareType<FromType>,InfinityType>::value, NumberType>
                to_ub(FromType&& f) noexcept
        {
            return std::forward<FromType>(f);
        }

        IVARP_HD static NumberType to_lb(InfinityType) noexcept {
            return -impl::inf_value<NumberType>();
        }

        IVARP_HD static NumberType to_ub(InfinityType) noexcept {
            return impl::inf_value<NumberType>();
        }

        NumberType m_lb, m_ub;
        bool m_undefined;
    };

    template<typename NumberType> static inline
    std::enable_if_t<!std::is_floating_point<NumberType>::value, std::ostream&>
        operator<<(std::ostream& o, const Interval<NumberType>& i)
    {
        if(i.possibly_undefined()) {
            o << '~';
        }
		o << '[';
		if (i.finite_lb()) {
			o << lb(i);
		} else {
			o << "-inf";
		}
		o << ", ";
		if (i.finite_ub()) {
			o << ub(i);
		} else {
			o << "inf";
		}
		o << ']';
		
        return o;
    }

    template<typename NumberType>
        static inline IVARP_HD bool is_finite(const Interval<NumberType>& n) noexcept
    {
        return n.is_finite();
    }

    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(typename IntervalType), IntervalType,
        static inline IntervalType mark_defined(IntervalType iv) noexcept(IsCudaNumber<IntervalType>::value) {
            iv.set_undefined(false);
            return ivarp::move(iv);
        }
    )

    static_assert(std::is_trivial<IFloat>::value, "IFloat should be trivial (copyable to devices)!");
    static_assert(std::is_trivial<IDouble>::value, "IDouble should be trivial (copyable to devices)!");

    template<typename IntervalType> struct IntervalLexicographicalCompare {
        IVARP_HD_OVERLOAD_ON_CUDA_NT(IntervalType,
            bool operator()(const IntervalType& i1, const IntervalType& i2) const noexcept
            {
                int lb_comp;
                if(!i1.finite_lb()) {
                    lb_comp = i2.finite_lb() ? -1 : 0;
                } else {
                    if(i2.finite_lb()) {
                        lb_comp = i1.lb() < i2.lb()  ? -1 :
                                  i1.lb() == i2.lb() ?  0 : 1;
                    } else {
                        lb_comp = 1;
                    }
                }
                if(lb_comp < 0) {
                    return true;
                }
                if(lb_comp > 0) {
                    return false;
                }
                if(!i1.finite_ub()) {
                    return false;
                } else {
                    return !i2.finite_ub() || i1.ub() < i2.ub();
                }
            }
        )
    };

    template<typename IntervalType> struct CuboidLexicographicalCompare {
        using IComp = IntervalLexicographicalCompare<IntervalType>;
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(std::size_t N),
            IntervalType,
            bool operator()(const Array<IntervalType, N>& c1, const Array<IntervalType, N>& c2) const noexcept {
                for(std::size_t i = 0; i < N; ++i) {
                    if(IComp{}(c1[i], c2[i])) {
                        return true;
                    }
                    if(IComp{}(c2[i], c1[i])) {
                        return false;
                    }
                }
                return false;
            }
        )
    };
}
