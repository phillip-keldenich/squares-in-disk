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
// Created by Phillip Keldenich on 2019-10-01.
//

#pragma once

#include "ivarp/math.hpp"
#include "interval_sqrt.hpp"
#include "interval_asin.hpp"
#include "interval_acos.hpp"

namespace ivarp {
namespace impl {
    /// Transcendental functions for intervals of builtin types.
    extern IVARP_EXPORTED_SYMBOL IFloat builtin_interval_sin(IFloat x);
    extern IVARP_EXPORTED_SYMBOL IDouble builtin_interval_sin(IDouble x);
    extern IVARP_EXPORTED_SYMBOL IFloat builtin_interval_cos(IFloat x);
    extern IVARP_EXPORTED_SYMBOL IDouble builtin_interval_cos(IDouble x);
    extern IVARP_EXPORTED_SYMBOL IFloat builtin_interval_exp(IFloat x);
    extern IVARP_EXPORTED_SYMBOL IDouble builtin_interval_exp(IDouble x);

    /// Irrational functions for intervals of rationals.
    extern IVARP_EXPORTED_SYMBOL IRational rational_interval_sin(const IRational& x, unsigned precision);
    extern IVARP_EXPORTED_SYMBOL IRational rational_interval_cos(const IRational& x, unsigned precision);
    extern IVARP_EXPORTED_SYMBOL IRational rational_interval_exp(const IRational& x, unsigned precision);

    /// Overloaded sine evaluation function forwarding to the corresponding functions for the given number type.
    template<typename Context, typename FloatType> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, FloatType> sin(FloatType x) noexcept
    {
        return IVARP_NOCUDA_USE_STD sin(x);
    }

    template<typename Context, typename FloatType> static inline
        std::enable_if_t<std::is_floating_point<FloatType>::value, Interval<FloatType>> sin(Interval<FloatType> x) noexcept
    {
        return impl::builtin_interval_sin(x);
    }

    template<typename Context> static inline IRational sin(const IRational& r) {
        return impl::rational_interval_sin(r, Context::irrational_precision);
    }

    /// Overloaded cosine evaluation function forwarding to the corresponding functions for the given number type.
    template<typename Context, typename FloatType> static inline IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, FloatType> cos(FloatType x) noexcept
    {
        return IVARP_NOCUDA_USE_STD cos(x);
    }

    template<typename Context, typename FloatType> static inline
        std::enable_if_t<std::is_floating_point<FloatType>::value, Interval<FloatType>> cos(Interval<FloatType> x) noexcept
    {
        return impl::builtin_interval_cos(x);
    }

    template<typename Context> static inline IRational cos(const IRational& r) {
        return impl::rational_interval_cos(r, Context::irrational_precision);
    }

    /// Overloaded exp evaluation function that forwards to the corresponding functions for the given number type.
    template<typename Context, typename FloatType> IVARP_HD
        std::enable_if_t<std::is_floating_point<FloatType>::value, FloatType> exp(FloatType x) noexcept
    {
        return IVARP_NOCUDA_USE_STD exp(x);
    }

    template<typename Context, typename FloatType>
        std::enable_if_t<std::is_floating_point<FloatType>::value, FloatType> exp(Interval<FloatType> x) noexcept
    {
        return impl::builtin_interval_exp(x);
    }

    template<typename Context> static inline IRational exp(const IRational& r) {
        return impl::rational_interval_exp(r, Context::irrational_precision);
    }
}

    // ---------------- UNARY MATH FUNCTIONS ----------------
    struct IrrationalTag {};
    template<typename Tag> using IsIrrationalTag = std::is_convertible<BareType<Tag>, IrrationalTag>;

    /// Tag for sine.
    struct MathSinTag  : IrrationalTag {
        static const char* name() noexcept {
            return "sin";
        }

        struct EvalBounds {
            template<typename B> using Eval = fixed_point_bounds::SinEvalBounds<B>;
        };

        template<typename Context, typename NumberType>
            static inline NumberType eval(const NumberType& n)
        {
            return ::ivarp::impl::sin<Context>(n);
        }
    };

    /// Tag for cosine.
    struct MathCosTag  : IrrationalTag {
        static const char* name() noexcept {
            return "cos";
        }

        struct EvalBounds {
            template<typename B> using Eval = fixed_point_bounds::CosEvalBounds<B>;
        };

        template<typename Context, typename NumberType>
            static inline NumberType eval(const NumberType& n)
        {
            return ::ivarp::impl::cos<Context>(n);
        }
    };

    /// Tag for square root.
    struct MathSqrtTag : IrrationalTag {
        static const char* name() noexcept {
            return "sqrt";
        }

        struct EvalBounds {
            template<typename B> using Eval = fixed_point_bounds::SqrtEvalBounds<B>;
        };

        struct BoundedEval {
            /// Interval sqrt becomes cheaper if we know the operand is non-negative at compile time.
            IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
                IVARP_TEMPLATE_PARAMS(typename Context, typename Bounds, typename NumberType), NumberType,
                static inline NumberType eval(const NumberType& x) {
                    return ::ivarp::bounded_sqrt<Context, Bounds>(x);
                }
            )
        };

        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
                static inline NumberType eval(const NumberType& n)
            {
                return ::ivarp::sqrt<Context>(n);
            }
        );
    };

    /// Tag for e ** x.
    struct MathExpTag  : IrrationalTag {
        static const char* name() noexcept {
            return "exp";
        }

        template<typename Context, typename NumberType>
            static inline NumberType eval(const NumberType& n)
        {
            return ::ivarp::impl::exp<Context>(n);
        }
    };

    /// Tag for arc sine.
    struct MathAsinTag : IrrationalTag {
        static const char* name() noexcept {
            return "asin";
        }

		IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
			IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
				static inline NumberType eval(const NumberType& n)
			{
				return ::ivarp::impl::interval_asin<Context, fixed_point_bounds::Unbounded>(n);
			}
		);

        struct BoundedEval {
            /// Interval arc sine becomes cheaper if we know the operand is between -1 and 1.
            IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
                IVARP_TEMPLATE_PARAMS(typename Context, typename Bounds, typename NumberType), NumberType,
                    static inline NumberType eval(const NumberType& n)
                {
                    return ::ivarp::impl::interval_asin<Context, Bounds>(n);
                }
            )
        };
    };

	/// Tag for arc cosine.
	struct MathAcosTag : IrrationalTag {
		static const char* name() noexcept {
			return "acos";
		}

		IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
			IVARP_TEMPLATE_PARAMS(typename Context, typename NumberType), NumberType,
				static inline NumberType eval(const NumberType& n)
			{
				return ::ivarp::impl::interval_acos<Context, fixed_point_bounds::Unbounded>(n);
			}
		);

		struct BoundedEval {
			/// Interval arc cosine becomes cheaper if we know the operand is between -1 and 1.
			IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
                IVARP_TEMPLATE_PARAMS(typename Context, typename Bounds, typename NumberType), NumberType,
                    static inline NumberType eval(const NumberType& n)
                {
                    return ::ivarp::impl::interval_acos<Context, Bounds>(n);
                }
            )
		};
	};

    template<typename A> using MathSin = MathUnary<MathSinTag, A>;
    template<typename A> using MathCos = MathUnary<MathCosTag, A>;
    template<typename A> using MathSqrt = MathUnary<MathSqrtTag, A>;
    template<typename A> using MathExp = MathUnary<MathExpTag, A>;
    template<typename A> using MathAsin = MathUnary<MathAsinTag, A>;
	template<typename A> using MathAcos = MathUnary<MathAcosTag, A>;

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathSin<BareType<A1>>>
            sin(A1&& arg)
    {
        return MathSin<BareType<A1>>(std::forward<A1>(arg));
    }

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathAsin<BareType<A1>>>
            asin(A1&& arg)
    {
        return MathAsin<BareType<A1>>(std::forward<A1>(arg));
    }

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathAcos<BareType<A1>>>
            acos(A1&& arg)
    {
        return MathAcos<BareType<A1>>(std::forward<A1>(arg));
    }

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathCos<BareType<A1>>>
            cos(A1&& arg)
    {
        return MathCos<BareType<A1>>(std::forward<A1>(arg));
    }

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathSqrt<BareType<A1>>>
            sqrt(A1&& arg)
    {
        return MathSqrt<BareType<A1>>(std::forward<A1>(arg));
    }

    template<typename A1> static constexpr inline
        std::enable_if_t<IsMathExpr<A1>::value, MathExp<BareType<A1>>>
            exp(A1&& arg)
    {
        return MathExp<BareType<A1>>(std::forward<A1>(arg));
    }
}
