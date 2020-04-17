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
    template<typename T, std::int64_t LB, std::int64_t UB, bool DefDefined> struct BoundedQ {
        static constexpr bool defined = DefDefined;
        static constexpr std::int64_t lb = LB;
        static constexpr std::int64_t ub = UB;
        T value;
    };

    template<typename T1, typename T2> using BQResultType =
        std::conditional_t<std::is_same<T1, IRational>::value || std::is_same<T2, IRational>::value,
                           IRational, Rational>;

    template<typename T1, std::int64_t LB, std::int64_t UB, bool D1>
        static inline auto operator-(const BoundedQ<T1, LB, UB, D1>& q)
    {
        return BoundedQ<T1, -UB, -LB, D1>{-q.value};
    }

    template<typename T1, std::int64_t LB1, std::int64_t UB1, bool D1,
             typename T2, std::int64_t LB2, std::int64_t UB2, bool D2>
        static inline auto operator+(const BoundedQ<T1, LB1, UB1, D1>& q1, const BoundedQ<T2, LB2, UB2, D2>& q2)
    {
        return BoundedQ<BQResultType<T1,T2>, fp_iv_add_lb(LB1, LB2), fp_iv_add_ub(UB1, UB2), D1 && D2>{
            q1.value + q2.value
        };
    }

    template<typename T1, std::int64_t LB1, std::int64_t UB1, bool D1,
             typename T2, std::int64_t LB2, std::int64_t UB2, bool D2>
        static inline auto operator-(const BoundedQ<T1, LB1, UB1, D1>& q1, const BoundedQ<T2, LB2, UB2, D2>& q2)
    {
        return BoundedQ<BQResultType<T1,T2>, fp_iv_add_lb(LB1, -UB2), fp_iv_add_ub(UB1, -LB2), D1 && D2>{
            q1.value - q2.value
        };
    }

    template<typename T1, std::int64_t LB1, std::int64_t UB1, bool D1,
             typename T2, std::int64_t LB2, std::int64_t UB2, bool D2>
        static inline auto operator*(const BoundedQ<T1, LB1, UB1, D1>& q1, const BoundedQ<T2, LB2, UB2, D2>& q2)
    {
        return BoundedQ<BQResultType<T1,T2>, fp_iv_mul_lb(LB1,UB1,LB2,UB2),
                        fp_iv_mul_ub(LB1,UB1,LB2,UB2), D1&&D2>
            { q1.value * q2.value };
    }

    template<typename T1, std::int64_t LB1, std::int64_t UB1, bool D1,
             typename T2, std::int64_t LB2, std::int64_t UB2, bool D2>
        static inline auto operator/(const BoundedQ<T1, LB1, UB1, D1>& q1, const BoundedQ<T2, LB2, UB2, D2>& q2)
    {
        static constexpr bool defined_type = !std::is_same<T1, IRational>::value && !std::is_same<T2, IRational>::value;

        return BoundedQ<BQResultType<T1,T2>, fp_iv_div_lb(LB1,UB1,LB2,UB2),
                        fp_iv_div_ub(LB1,UB1,LB2,UB2),
                        D1 && D2 && (defined_type || nonzero(LB2,UB2))>
            { q1.value / q2.value };
    }
}
}

namespace ivarp {
    template<std::int64_t LB, std::int64_t UB> using BoundedRational =
        fixed_point_bounds::BoundedQ<Rational, LB, UB, true>;
    template<std::int64_t LB, std::int64_t UB, bool DefDefined> using BoundedIRational =
        fixed_point_bounds::BoundedQ<IRational, LB, UB, DefDefined>;
}
