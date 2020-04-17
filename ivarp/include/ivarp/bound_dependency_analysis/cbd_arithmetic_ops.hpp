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
// Created by Phillip Keldenich on 28.01.20.
//

#pragma once

namespace ivarp {
namespace impl {
    /**
     * Implementation of bound dependency analysis for unary operator-.
     * @tparam ArgIndex
     * @tparam Child
     * @tparam ChildResult
     */
    template<std::size_t ArgIndex, typename Child, typename ChildResult>
        struct ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathUnary<MathOperatorTagUnaryMinus, Child>, ChildResult>
    {
        struct Type {
            static constexpr bool lb_depends_on_lb = ChildResult::ub_depends_on_lb;
            static constexpr bool lb_depends_on_ub = ChildResult::ub_depends_on_ub;
            static constexpr bool ub_depends_on_lb = ChildResult::lb_depends_on_lb;
            static constexpr bool ub_depends_on_ub = ChildResult::lb_depends_on_ub;
        };
    };

    /**
     * Implementation of bound dependency analysis for binary operator+.
     * @tparam ArgIndex
     * @tparam C1
     * @tparam C2
     * @tparam CR1
     * @tparam CR2
     */
    template<std::size_t ArgIndex, typename C1, typename C2, typename CR1, typename CR2>
        struct ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathBinary<MathOperatorTagAdd, C1, C2>, CR1, CR2>
    {
        struct Type {
            static constexpr bool lb_depends_on_lb = CR1::lb_depends_on_lb || CR2::lb_depends_on_lb;
            static constexpr bool lb_depends_on_ub = CR1::lb_depends_on_ub || CR2::lb_depends_on_ub;
            static constexpr bool ub_depends_on_lb = CR1::ub_depends_on_lb || CR2::ub_depends_on_lb;
            static constexpr bool ub_depends_on_ub = CR1::ub_depends_on_ub || CR2::ub_depends_on_ub;
        };
    };

    /**
     * Implementation of bound dependency analysis for binary operator-.
     * @tparam ArgIndex
     * @tparam C1
     * @tparam C2
     * @tparam CR1
     * @tparam CR2
     */
    template<std::size_t ArgIndex, typename C1, typename C2, typename CR1, typename CR2>
        struct ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathBinary<MathOperatorTagSub, C1, C2>, CR1, CR2>
    {
        struct Type {
            static constexpr bool lb_depends_on_lb = CR1::lb_depends_on_lb || CR2::ub_depends_on_lb;
            static constexpr bool lb_depends_on_ub = CR1::lb_depends_on_ub || CR2::ub_depends_on_ub;
            static constexpr bool ub_depends_on_lb = CR1::ub_depends_on_lb || CR2::lb_depends_on_lb;
            static constexpr bool ub_depends_on_ub = CR1::ub_depends_on_ub || CR2::lb_depends_on_ub;
        };
    };

    /**
     * Implementation of bound dependency analysis for binary operator*.
     * @tparam ArgIndex
     * @tparam C1
     * @tparam C2
     * @tparam CR1
     * @tparam CR2
     */
     template<std::size_t ArgIndex, typename C1, typename C2, typename CR1, typename CR2>
        struct ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathBinary<MathOperatorTagMul, C1, C2>, CR1, CR2>
    {
    private:
        static constexpr int sgn1 =
                (C1::lb >= 0) ? 1 :
                (C1::ub <= 0) ? -1 : 0;

        static constexpr int sgn2 =
                (C2::lb >= 0) ? 1 :
                (C2::ub <= 0) ? -1 : 0;

        static constexpr BoundID lb_uses_bounds1 =
                (sgn2 ==  1) ? BoundID::LB :
                (sgn2 == -1) ? BoundID::UB :
                (sgn1 ==  1) ? BoundID::UB :
                (sgn1 == -1) ? BoundID::LB : BoundID::BOTH;

        static constexpr BoundID ub_uses_bounds1 =
                (sgn2 ==  1) ? BoundID::UB :
                (sgn2 == -1) ? BoundID::LB :
                (sgn1 ==  1) ? BoundID::UB :
                (sgn1 == -1) ? BoundID::LB : BoundID::BOTH;

        static constexpr BoundID lb_uses_bounds2 =
                (sgn1 ==  1) ? BoundID::LB :
                (sgn1 == -1) ? BoundID::UB :
                (sgn2 ==  1) ? BoundID::UB :
                (sgn2 == -1) ? BoundID::LB : BoundID::BOTH;

        static constexpr BoundID ub_uses_bounds2 =
                (sgn1 ==  1) ? BoundID::UB :
                (sgn1 == -1) ? BoundID::LB :
                (sgn2 ==  1) ? BoundID::UB :
                (sgn2 == -1) ? BoundID::LB : BoundID::BOTH;

        static constexpr BoundDependencies bdeps1 = BoundDependencies::from_type<CR1>();
        static constexpr BoundDependencies bdeps2 = BoundDependencies::from_type<CR2>();
        static constexpr BoundDependencies result =
            BoundDependencies::computation_uses(lb_uses_bounds1, ub_uses_bounds1, bdeps1,
                                                lb_uses_bounds2, ub_uses_bounds2, bdeps2);

    public:
        struct Type {
            static constexpr bool lb_depends_on_lb = result.lb_depends_on_lb;
            static constexpr bool lb_depends_on_ub = result.lb_depends_on_ub;
            static constexpr bool ub_depends_on_lb = result.ub_depends_on_lb;
            static constexpr bool ub_depends_on_ub = result.ub_depends_on_ub;
        };
    };

     /**
     * Implementation of bound dependency analysis for binary operator/.
     * @tparam ArgIndex
     * @tparam C1
     * @tparam C2
     * @tparam CR1
     * @tparam CR2
     */
    template<std::size_t ArgIndex, typename C1, typename C2, typename CR1, typename CR2>
        struct ComputeBoundDepsMetaEvalTagImpl<ArgIndex, MathBinary<MathOperatorTagDiv, C1, C2>, CR1, CR2>
    {
     private:
        static constexpr int sgn1 =
                (C1::lb >= 0) ? 1 :
                (C1::ub <= 0) ? -1 : 0;

        static constexpr int sgn2 =
                (C2::lb > 0) ? 1 :
                (C2::ub < 0) ? -1 : 0;

        static constexpr BoundID lb_uses_bounds1 =
                (sgn2 ==  1) ? BoundID::LB :
                (sgn2 == -1) ? BoundID::UB : BoundID::BOTH;

        static constexpr BoundID ub_uses_bounds1 =
                (sgn2 ==  1) ? BoundID::UB :
                (sgn2 == -1) ? BoundID::LB : BoundID::BOTH;

        static constexpr BoundID lb_uses_bounds2 =
                (sgn2 ==  0) ? BoundID::BOTH : // divisor is mixed
                (sgn1 ==  1) ? BoundID::UB :   // dividend is positive
                (sgn1 == -1) ? BoundID::LB :   // dividend is negative
                (sgn2 ==  1) ? BoundID::LB :   // dividend is mixed, divisor positive
                               BoundID::UB;    // dividend is mixed, divisor negative

        static constexpr BoundID ub_uses_bounds2 =
                (sgn2 ==  0) ? BoundID::BOTH :  // divisor is mixed
                (sgn1 ==  1) ? BoundID::LB :    // dividend is positive
                (sgn1 == -1) ? BoundID::UB :    // dividend is negative
                (sgn2 ==  1) ? BoundID::LB :    // dividend mixed, divisor positive
                               BoundID::UB;     // dividend mixed, divisor negative

        static constexpr BoundDependencies bdeps1 = BoundDependencies::from_type<CR1>();
        static constexpr BoundDependencies bdeps2 = BoundDependencies::from_type<CR2>();
        static constexpr BoundDependencies result =
            BoundDependencies::computation_uses(lb_uses_bounds1, ub_uses_bounds1, bdeps1,
                                                lb_uses_bounds2, ub_uses_bounds2, bdeps2);

    public:
        struct Type {
            static constexpr bool lb_depends_on_lb = result.lb_depends_on_lb;
            static constexpr bool lb_depends_on_ub = result.lb_depends_on_ub;
            static constexpr bool ub_depends_on_lb = result.ub_depends_on_lb;
            static constexpr bool ub_depends_on_ub = result.ub_depends_on_ub;
        };
    };
}
}
