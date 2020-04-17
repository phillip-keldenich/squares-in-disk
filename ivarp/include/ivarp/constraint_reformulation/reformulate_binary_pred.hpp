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
// Created by Phillip Keldenich on 11.11.19.
//

#pragma once

namespace ivarp {
    namespace impl {
        template<typename Tag, typename Left, typename Right, typename Arg,
                 bool IsTermTag = TypeIn<Tag, BinaryMathPredGEQ, BinaryMathPredLEQ,
                                         BinaryMathPredGT, BinaryMathPredLT, BinaryMathPredEQ>::value
        > struct CanReformulateBinaryPredIntoBound : std::false_type {};

        template<typename Tag, typename Left, typename Right, typename Arg>
            struct CanReformulateBinaryPredIntoBound<Tag, Left, Right, Arg, true>
        {
            static constexpr bool value = std::is_same<Left, Arg>::value ^ std::is_same<Right, Arg>::value;
        };

        template<typename Tag, typename Left, typename Right, typename Arg,
                 bool ArgIsLeft = std::is_same<Left, Arg>::value,
                 bool ArgIsRight = std::is_same<Right, Arg>::value>
        struct ReformulateBinaryPredIntoBound;

        template<typename Tag, typename Left, typename Right, typename Arg>
            struct ReformulateBinaryPredIntoBound<Tag, Left, Right, Arg, true, false>
        {
            using BoundFunctionType = Right;

            static BoundFunctionType bound_function(const BinaryMathPred<Tag,Left,Right>& constraint) {
                return constraint.arg2;
            }

            template<typename Context, typename ArgArray>
            static BoundDirection bound_direction(const BinaryMathPred<Tag,Left,Right>& /*constraint*/,
                                                  const Right& /*bound_fn*/, const ArgArray& /*args*/) noexcept
            {
                constexpr bool is_leq = TypeIn<Tag, BinaryMathPredEQ, BinaryMathPredLT, BinaryMathPredLEQ>::value;
                constexpr bool is_geq = TypeIn<Tag, BinaryMathPredEQ, BinaryMathPredGT, BinaryMathPredGEQ>::value;
                return (is_leq ? BoundDirection::LEQ : BoundDirection::NONE) |
                       (is_geq ? BoundDirection::GEQ : BoundDirection::NONE);
            }
        };

        template<typename Tag, typename Left, typename Right, typename Arg>
            struct ReformulateBinaryPredIntoBound<Tag, Left, Right, Arg, false, true>
        {
            using BoundFunctionType = Left;

            static BoundFunctionType bound_function(const BinaryMathPred<Tag,Left,Right>& constraint) {
                return constraint.arg1;
            }

            template<typename Context, typename ArgArray>
            static BoundDirection bound_direction(const BinaryMathPred<Tag,Left,Right>& /*constraint*/,
                                                  const Left& /*bound_fn*/, const ArgArray& /*args*/) noexcept
            {
                constexpr bool is_geq = TypeIn<Tag, BinaryMathPredEQ, BinaryMathPredLT, BinaryMathPredLEQ>::value;
                constexpr bool is_leq = TypeIn<Tag, BinaryMathPredEQ, BinaryMathPredGT, BinaryMathPredGEQ>::value;
                return (is_leq ? BoundDirection::LEQ : BoundDirection::NONE) |
                       (is_geq ? BoundDirection::GEQ : BoundDirection::NONE);
            }
        };
    }

    template<typename Tag, typename Left, typename Right, typename Arg>
        struct CanReformulateIntoBound<BinaryMathPred<Tag,Left,Right>, Arg> :
            impl::CanReformulateBinaryPredIntoBound<Tag,Left,Right,Arg>
    {};

    template<typename Tag, typename Left, typename Right, typename Arg>
        struct ReformulateIntoBound<BinaryMathPred<Tag,Left,Right>, Arg> :
            impl::ReformulateBinaryPredIntoBound<Tag,Left,Right,Arg>
    {};
}
