//
// Created by Phillip Keldenich on 22.04.20.
//

#pragma once

#ifdef __CUDACC__
#include "ivarp_cuda/interval_asin.cuh"
#endif

namespace ivarp {
namespace impl {
    IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
        IVARP_TEMPLATE_PARAMS(typename Bounds, typename Domain, typename NumberType), NumberType,
        static inline bool fn_domain_handle_bounds(Interval<NumberType>& arg, NumberType dom_lb, NumberType dom_ub) {
            bool p_undef = arg.possibly_undefined();
            if(constexpr_fn<std::int64_t, Bounds::lb>() < constexpr_fn<std::int64_t, Domain::lb>() && arg.above_lb(dom_lb)) {
                p_undef = true;
                arg.set_lb(dom_lb);
            }
            if(constexpr_fn<std::int64_t, Bounds::ub>() > constexpr_fn<std::int64_t, Domain::ub>() && arg.below_ub(dom_ub)) {
                p_undef = true;
                arg.set_ub(dom_ub);
            }
            return p_undef;
        }
    )

    struct SinRangeDomain {
        static constexpr std::int64_t lb = fixed_point_bounds::int_to_fp(-1);
        static constexpr std::int64_t ub = fixed_point_bounds::int_to_fp(1);
    };

    extern IVARP_H IVARP_EXPORTED_SYMBOL float cpu_asin_rd(float x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL float cpu_asin_ru(float x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL double cpu_asin_rd(double x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL double cpu_asin_ru(double x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL IRational rational_interval_asin(const IRational& x, unsigned precision);

    template<typename Context, typename Bounds, typename NumberType,
             std::enable_if_t<std::is_floating_point<NumberType>::value, int> = 0>
    static inline IVARP_HD Interval<NumberType> interval_asin(Interval<NumberType> x) noexcept
    {
        bool p_undef = fn_domain_handle_bounds<Bounds, SinRangeDomain>(x, NumberType(-1), NumberType(1));
#ifdef __CUDA_ARCH__
        return Interval<NumberType>{cuda_asin_rd(x.lb()), cuda_asin_ru(x.ub()), p_undef};
#else
        return Interval<NumberType>{cpu_asin_rd(x.lb()), cpu_asin_ru(x.ub()), p_undef};
#endif
    }

    template<typename Context, typename Bounds>
    static inline IVARP_H IRational interval_asin(const IRational& x) {
        return rational_interval_asin(x, Context::irrational_precision);
    }

    template<typename Context, typename Bounds>
    static inline IVARP_HD float interval_asin(float x) noexcept {
#ifdef __CUDA_ARCH__
        return asinf(x);
#else
        return std::asin(x);
#endif
    }

    template<typename Context, typename Bounds>
    static inline IVARP_HD double interval_asin(double x) noexcept {
        return IVARP_NOCUDA_USE_STD asin(x);
    }
}
}
