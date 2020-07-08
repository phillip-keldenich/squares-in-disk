//
// Created by Phillip Keldenich on 22.04.2020.
//

#pragma once

#ifdef __CUDACC__
#include "ivarp_cuda/interval_acos.cuh"
#endif

namespace ivarp {
namespace impl {
	extern IVARP_H IVARP_EXPORTED_SYMBOL float cpu_acos_rd(float x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL float cpu_acos_ru(float x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL double cpu_acos_rd(double x) noexcept;
    extern IVARP_H IVARP_EXPORTED_SYMBOL double cpu_acos_ru(double x) noexcept;
	extern IVARP_H IVARP_EXPORTED_SYMBOL IRational rational_interval_acos(const IRational& x, unsigned precision);

	template<typename Context, typename Bounds, typename NumberType,
             std::enable_if_t<std::is_floating_point<NumberType>::value, int> = 0>
    static inline IVARP_HD Interval<NumberType> interval_acos(Interval<NumberType> x) noexcept
    {
        bool p_undef = fn_domain_handle_bounds<Bounds, SinRangeDomain>(x, NumberType(-1), NumberType(1));
#ifdef __CUDA_ARCH__
        return Interval<NumberType>{cuda_acos_rd(x.ub()), cuda_acos_ru(x.lb()), p_undef};
#else
        return Interval<NumberType>{cpu_acos_rd(x.ub()), cpu_acos_ru(x.lb()), p_undef};
#endif
    }

    template<typename Context, typename Bounds>
    static inline IVARP_H IRational interval_acos(const IRational& x) {
        return rational_interval_acos(x, Context::irrational_precision);
    }

    template<typename Context, typename Bounds>
    static inline IVARP_HD float interval_acos(float x) noexcept {
#ifdef __CUDA_ARCH__
        return acosf(x);
#else
        return std::acos(x);
#endif
    }

    template<typename Context, typename Bounds>
    static inline IVARP_HD double interval_acos(double x) noexcept {
        return IVARP_NOCUDA_USE_STD acos(x);
    }
}
}

