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
// Created by Phillip Keldenich on 15.01.20.
//

#pragma once

#include "ivarp/math_fn.hpp"

namespace ivarp {
    template<typename MathArgType, typename LBType_, typename UBType_> struct VariableDescription {
        using Arg = MathArgType;
        using LBType = LBType_;
        using UBType = UBType_;
        constexpr static unsigned index = Arg::index;

        static_assert(NumArgs<LBType>::value <= index,
                      "Variable lower bound depends on variables/values of higher index!");
        static_assert(NumArgs<UBType>::value <= index,
                      "Variable upper bound depends on variables/values of higher index!");

        template<typename StringArg, typename LBT, typename UBT>
            IVARP_H VariableDescription(StringArg&& name, LBT&& lb, UBT&& ub) :
                name(std::forward<StringArg>(name)),
                lb(std::forward<LBT>(lb)),
                ub(std::forward<UBT>(ub))
        {}

        // these could be defaulted, but that causes CUDA warnings because it somehow assumes the defaulted version to be __host__ __device__
        IVARP_H VariableDescription(const VariableDescription& v) :
            name(v.name),
            lb(v.lb), ub(v.ub)
        {}
        IVARP_H VariableDescription(VariableDescription&& v) noexcept :
            name(std::move(v.name)),
            lb(std::move(v.lb)),
            ub(std::move(v.ub))
        {}

        std::string name;
        LBType lb;
        UBType ub;
    };

    namespace impl {
        template<typename T> struct IsVariableDescriptionImpl : std::false_type {};
        template<typename MAT, typename L, typename U> struct IsVariableDescriptionImpl<VariableDescription<MAT,L,U>> :
            std::true_type
        {};
    }
    template<typename T> using IsVariableDescription = impl::IsVariableDescriptionImpl<BareType<T>>;

    template<typename StringArg, typename LBT, typename UBT, typename ArgIndexType>
        static IVARP_H inline auto variable(const MathArg<ArgIndexType>& /*arg*/,
                                            StringArg&& name, LBT&& lb, UBT&& ub)
    {
        return VariableDescription<MathArg<ArgIndexType>, EnsureExpr<BareType<LBT>>, EnsureExpr<BareType<UBT>>>{
            std::forward<StringArg>(name), ensure_expr(std::forward<LBT>(lb)), ensure_expr(std::forward<UBT>(ub))
        };
    }

    template<typename MathArgType, typename ChildExpr> struct ValueDescription {
        using Arg = MathArgType;
        using Child = ChildExpr;
        using LBType = Child;
        using UBType = Child;
        constexpr static unsigned index = Arg::index;

        static_assert(NumArgs<Child>::value <= index, "Value depends on variables/values of higher index!");

        template<typename StringArg, typename ExprT>
            IVARP_H ValueDescription(StringArg&& name, ExprT&& expr) :
                name(std::forward<StringArg>(name)),
                expr(std::forward<ExprT>(expr))
        {}

        std::string name;
        Child       expr;
    };

    namespace impl {
        template<typename T> struct IsValueDescriptionImpl : std::false_type {};
        template<typename MAT, typename C> struct IsValueDescriptionImpl<ValueDescription<MAT,C>> :
            std::true_type
        {};
    }
    template<typename T> using IsValueDescription = impl::IsValueDescriptionImpl<BareType<T>>;
    template<typename T> using IsArgDescription = std::integral_constant<bool, IsVariableDescription<T>::value ||
                                                                               IsValueDescription<T>::value>;

    template<typename StringArg, typename Expr, typename ArgIndexType>
        static IVARP_H inline auto value(const MathArg<ArgIndexType>& /*arg*/, Expr&& expr, StringArg&& name)
    {
        return ValueDescription<MathArg<ArgIndexType>, EnsureExpr<BareType<Expr>>>{
            std::forward<StringArg>(name), ensure_expr(std::forward<Expr>(expr))
        };
    }
}
