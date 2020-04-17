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
// Created by Phillip Keldenich on 2019-10-02.
//

#pragma once

namespace ivarp {
    namespace impl {
        template<typename T1, typename T2, bool I1, bool I2> struct MaxRankImpl;

        // rule 1: Intervals have higher rank
        template<typename T1, typename T2> struct MaxRankImpl<T1,T2,true,false> {
            using type = T1;
        };
        template<typename T1, typename T2> struct MaxRankImpl<T1,T2,false,true> {
            using type = T2;
        };

        // for two intervals, use an interval with the number type of higher rank
        template<typename T1, typename T2> struct MaxRankImpl<T1,T2,true,true> {
            using type = Interval<typename MaxRankImpl<typename T1::NumberType, typename T2::NumberType, false, false>::type>;
        };

        // if we only have one type, choose that one.
        template<typename T1> struct MaxRankImpl<T1,T1,false,false> {
            using type = T1;
        };

        template<> struct MaxRankImpl<float, double, false, false> {
            using type = double;
        };

        template<> struct MaxRankImpl<double, float, false, false> {
            using type = double;
        };

        template<> struct MaxRankImpl<Rational, double, false, false> {
            using type = Rational;
        };

        template<> struct MaxRankImpl<Rational, float, false, false> {
            using type = Rational;
        };

        template<> struct MaxRankImpl<double, Rational, false, false> {
            using type = Rational;
        };

        template<> struct MaxRankImpl<float, Rational, false, false> {
            using type = Rational;
        };

        // metafunction that returns the input type with highest rank.
        template<typename T1, typename T2> struct MaxRank {
        private:
            using Type1 = BareType<T1>;
            using Type2 = BareType<T2>;

        public:
            using type = typename MaxRankImpl<
                Type1, Type2,
                NumberTraits<Type1>::is_interval, NumberTraits<Type2>::is_interval
            >::type;
        };

        template<typename CurrentType, typename... Args> struct NumberTypePromotionImpl;
        template<typename CurrentType> struct NumberTypePromotionImpl<CurrentType> {
            using type = CurrentType;
        };
        template<typename CurrentType, typename A1, typename... Args> struct NumberTypePromotionImpl<CurrentType, A1, Args...> {
            using type = typename NumberTypePromotionImpl<typename MaxRank<CurrentType,A1>::type, Args...>::type;
        };
    }

    template<typename Arg1, typename... Args> struct NumberTypePromotion {
        using type = typename impl::NumberTypePromotionImpl<Arg1, Args...>::type;
    };

    namespace impl {
        template<typename TargetType> struct ConvertTo;

        template<> struct ConvertTo<float> {
            float convert(std::intmax_t i) noexcept { return static_cast<float>(i); }
            float convert(std::uintmax_t i) noexcept { return static_cast<float>(i); }
            float convert(float f)  noexcept { return f; }
            float convert(double d) noexcept { return static_cast<float>(d); }
            float convert(const Rational& r) { return static_cast<float>(r.get_d()); }
            float convert(IFloat f) noexcept { return f.lb(); }
            float convert(IDouble d) noexcept { return static_cast<float>(d.lb()); }
            float convert(const IRational& r) { return static_cast<float>(r.lb().get_d()); }
        };

        template<> struct ConvertTo<double> {
            double convert(std::intmax_t i) noexcept { return static_cast<double>(i); }
            double convert(std::uintmax_t i) noexcept { return static_cast<double>(i); }
            double convert(float f) noexcept { return f; }
            double convert(double d) noexcept { return d; }
            double convert(const Rational& r) { return r.get_d(); }
            double convert(IFloat f) noexcept { return f.lb(); }
            double convert(IDouble d) noexcept { return d.lb(); }
            double convert(const IRational& r) { return r.lb().get_d(); }
        };

        template<> struct ConvertTo<Rational> {
            Rational convert(std::intmax_t i) noexcept { return Rational(i); }
            Rational convert(std::uintmax_t i) noexcept { return Rational(i); }
            Rational convert(float f) { return Rational{f}; }
            Rational convert(double d) { return Rational{d}; }
            Rational convert(const Rational& r) { return r; }
            Rational convert(IFloat f) { return Rational{f.lb()}; }
            Rational convert(IDouble d) { return Rational{d.lb()}; }
            Rational convert(const IRational& r) { return r.lb(); }
        };

        /// Convert the given integer to a floating-point number, in case the int definitely fits.
        template<typename FloatType, typename IntType,
                 bool Fits=(std::numeric_limits<FloatType>::digits >= std::numeric_limits<IntType>::digits)>
            struct IntervalFromIntImpl
        {
            static inline Interval<FloatType> convert(IntType i) noexcept {
                auto f = static_cast<FloatType>(i);
                return {f,f};
            }
        };

        template<typename FloatType, typename IntType>
            struct IntervalFromIntImpl<FloatType,IntType,false>
        {
            // Silence warnings and work around potential UB for unsigned/large integer types.
            template<typename IT> static std::enable_if_t<std::is_signed<IT>::value, IT> abs_(IT i) noexcept {
                return i < 0 ? -i : i;
            }
            template<typename IT> static std::enable_if_t<std::is_unsigned<IT>::value, IT> abs_(IT i) noexcept {
                return i;
            }

            static inline Interval<FloatType> convert(IntType i) noexcept {
                IntType max_exact = IntType(1) << std::numeric_limits<FloatType>::digits;
                IntType absi = abs_(i);
                auto f = static_cast<FloatType>(i);
                if(BOOST_LIKELY(absi <= max_exact)) {
                    return {f,f};
                } else {
                    if(exact_less_than(f, i)) {
                        return {f,step_up(f)};
                    } else if(exact_less_than(i,f)) {
                        return {step_down(f),f};
                    } else {
                        return {f,f};
                    }
                }
            }
        };

        template<typename FloatType, typename IntType>
            static inline Interval<FloatType> interval_from_int(IntType i) noexcept
        {
            return IntervalFromIntImpl<FloatType,IntType>::convert(i);
        }

        extern IVARP_EXPORTED_SYMBOL IFloat ifloat_from_double(double d);
        extern IVARP_EXPORTED_SYMBOL IFloat ifloat_from_rational(const Rational& r);
        template<> struct ConvertTo<IFloat> {
            IFloat convert(std::intmax_t i) noexcept { return interval_from_int<float>(i); }
            IFloat convert(std::uintmax_t i) noexcept { return interval_from_int<float>(i); }
            IFloat convert(float f) noexcept { return {f,f}; }
            IFloat convert(double d) noexcept { return ifloat_from_double(d); }
            IFloat convert(const Rational& r) { return ifloat_from_rational(r); }
            IFloat convert(const IFloat& f) noexcept { return f; }
            IFloat convert(IDouble d) {
                return {
                    ifloat_from_double(d.lb()).lb(),
                    ifloat_from_double(d.ub()).ub(),
                    d.possibly_undefined()
                };
            }
            IFloat convert(const IRational& r) {
                return { ifloat_from_rational(r.lb()).lb(), ifloat_from_rational(r.ub()).ub(), r.possibly_undefined() };
            }
        };

        extern IVARP_EXPORTED_SYMBOL IDouble idouble_from_rational(const Rational& r);
        template<> struct ConvertTo<IDouble> {
            IDouble convert(std::intmax_t i) noexcept { return interval_from_int<double>(i); }
            IDouble convert(std::uintmax_t i) noexcept { return interval_from_int<double>(i); }
            IDouble convert(float f)  noexcept { return {f,f}; }
            IDouble convert(double d) noexcept { return {d,d}; }
            IDouble convert(const Rational& r) {
                return { idouble_from_rational(r).lb(), idouble_from_rational(r).ub()};
            }
            IDouble convert(IFloat f) noexcept { return {f.lb(),f.ub(),f.possibly_undefined()}; }
            IDouble convert(const IDouble& d) noexcept { return d; }
            IDouble convert(const IRational& r) {
                return {
                    idouble_from_rational(r.lb()).lb(),
                    idouble_from_rational(r.ub()).ub(),
                    r.possibly_undefined()
                };
            }
        };

        template<> struct ConvertTo<IRational> {
            template<typename FloatType>
                static IRational from_float_i(Interval<FloatType> i)
            {
                bool lbf = std::isfinite(lb(i));
                bool ubf = std::isfinite(ub(i));

                FloatType lbval = lbf ? lb(i) : -std::numeric_limits<FloatType>::max();
                FloatType ubval = ubf ? ub(i) : std::numeric_limits<FloatType>::max();

                return IRational{ lbval, ubval, i.possibly_undefined(), !lbf, !ubf };
            }

            IRational convert(std::intmax_t i) noexcept {
                return IRational{i};
            }
            IRational convert(std::uintmax_t i) noexcept {
                return IRational{i};
            }
            IRational convert(float f) {
                return IRational{Rational{f}};
            }
            IRational convert(double d) {
                return IRational{Rational{d}};
            }
            IRational convert(const Rational& r) {
                return IRational{r};
            }
            IRational convert(IFloat f) {
                return from_float_i(f);
            }
            IRational convert(IDouble f) {
                return from_float_i(f);
            }
            IRational convert(const IRational& r) {
                return r;
            }
        };
    }

    template<typename TargetType, typename SourceType> static inline
        std::enable_if_t<!std::is_integral<SourceType>::value, TargetType> convert_number(const SourceType& source)
    {
        return impl::ConvertTo<TargetType>{}.convert(source);
    }

    template<typename TargetType, typename SourceType> static inline
        std::enable_if_t<std::is_integral<SourceType>::value && std::is_signed<SourceType>::value, TargetType>
            convert_number(const SourceType& source)
    {
        return impl::ConvertTo<TargetType>{}.convert(std::intmax_t(source));
    }

    template<typename TargetType, typename SourceType> static inline
        std::enable_if_t<std::is_integral<SourceType>::value && !std::is_signed<SourceType>::value, TargetType>
            convert_number(const SourceType& source)
    {
        return impl::ConvertTo<TargetType>{}.convert(std::uintmax_t(source));
    }

    template<typename TargetType, typename S, std::int64_t LB, std::int64_t UB, bool DefDefined>
        static inline TargetType convert_number(const fixed_point_bounds::BoundedQ<S, LB, UB, DefDefined>& source)
    {
        return convert_number<TargetType>(source.value);
    }
}
