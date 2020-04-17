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
// Created by Phillip Keldenich on 16.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename Child, typename Arg, bool CanReformulateChild = CanReformulateIntoBound<Child,Arg>::value>
        struct CanReformulateNegationIntoBound : std::false_type {};

    template<typename Child, typename Arg> struct CanReformulateNegationIntoBound<Child,Arg,true> : std::true_type {};

    template<typename Child, typename Arg> struct ReformulateNegationIntoBound {
    private:
        using ChildReformulator = ReformulateIntoBound<Child,Arg>;

    public:
        using BoundFunctionType = typename ChildReformulator::BoundFunctionType;

        static BoundFunctionType bound_function(const UnaryMathPred<UnaryMathPredNotTag,Child>& constraint) {
            return ChildReformulator::bound_function(constraint.arg);
        }

        template<typename Context, typename ArgArray> static BoundDirection
            bound_direction(const UnaryMathPred<UnaryMathPredNotTag,Child>& constraint,
                            const BoundFunctionType& bfn, const ArgArray& args)
        {
            BoundDirection bd = ChildReformulator::template bound_direction<Context>(constraint.arg, bfn, args);
            if(bd == BoundDirection::LEQ) {
                return BoundDirection::GEQ;
            } else if(bd == BoundDirection::GEQ) {
                return BoundDirection::LEQ;
            } else {
                return BoundDirection::NONE;
            }
        }
    };
}

    template<typename Child, typename Arg>
        struct CanReformulateIntoBound<UnaryMathPred<UnaryMathPredNotTag,Child>, Arg> :
            impl::CanReformulateNegationIntoBound<Child,Arg>
    {};

    template<typename Child, typename Arg> struct ReformulateIntoBound<UnaryMathPred<UnaryMathPredNotTag,Child>, Arg> :
        impl::ReformulateNegationIntoBound<Child,Arg>
    {};
}
