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
// Created by Phillip Keldenich on 29.10.19.
//

#pragma once

#include "ivarp/tuple.hpp"
#include <string>

namespace ivarp {
    template<typename ArgType, typename LBType, typename UBType, int Subdivision>
    class Variable {
    public:
        static_assert(IsMathExpr<LBType>::value, "Lower bound must be an expression.");
        static_assert(IsMathExpr<UBType>::value, "Upper bound must be an expression.");
        static_assert(NumArgs<LBType>::value <= ArgType::index,
                      "Variable lower bound depends on a variable with higher index!");
        static_assert(NumArgs<UBType>::value <= ArgType::index,
                      "Variable upper bound depends on a variable with higher index!");

        template<typename StringArg, typename LBT, typename UBT>
            Variable(const ArgType& arg, StringArg&& name, LBT&& lb, UBT&& ub) :
                arg(arg), m_name(std::forward<StringArg>(name)), lb(std::forward<LBT>(lb)), ub(std::forward<UBT>(ub))
        {}

        static constexpr bool is_dynamic = Subdivision < 0;
        static constexpr int num_subintervals = is_dynamic ? -Subdivision : Subdivision;

        template<typename Context, typename ArrayType> typename Context::NumberType
            compute_bounds(const Context& /*ctx*/, const ArrayType& args) const
        {
            return Context::NumberType::combine_bounds(lb.template array_evaluate<Context>(args),
                                                       ub.template array_evaluate<Context>(args));
        }

        const std::string& name() const noexcept {
            return m_name;
        }

        struct InitialCompileTimeBounds {
            template<typename PrevVarBoundTuple> struct Bounds {
                static constexpr std::int64_t lb = CompileTimeBounds<LBType, PrevVarBoundTuple>::lb;
                static constexpr std::int64_t ub = CompileTimeBounds<UBType, PrevVarBoundTuple>::ub;
                static_assert(lb <= ub, "Initial bounds of variable are empty (proof successful at compile time?)!");
            };
        };

    private:
        ArgType arg;
        std::string m_name;
        LBType lb;
        UBType ub;
    };

    template<typename T> struct IsVariable {
        static constexpr bool value = false;
    };
    template<typename A, typename L, typename U, int S> struct IsVariable<Variable<A,L,U,S>> {
        static constexpr bool value = true;
    };

    template<int Subdivision, typename T, typename StringArg, typename LBType, typename UBType>
        Variable<MathArg<T>, EnsureExpr<LBType>, EnsureExpr<UBType>, Subdivision>
            variable(const MathArg<T>& arg, StringArg&& name, LBType&& lb, UBType&& ub)
    {
        return {arg, std::forward<StringArg>(name), ensure_expr(lb), ensure_expr(ub)};
    }

    namespace impl {
        template<std::size_t Count, typename... VarExprs> struct CountDynamicVars {
            static constexpr unsigned value = Count;
        };

        template<std::size_t Result, typename... VarExprs> struct ComputeInitialDynamicCuboids {
            static constexpr std::size_t value = Result;
        };

        template<std::size_t Result, typename VarExpr, typename... VarExprs>
            struct ComputeInitialDynamicCuboids<Result, VarExpr, VarExprs...>
        {
            static constexpr std::size_t value = VarExpr::is_dynamic ?
                    ComputeInitialDynamicCuboids<Result * VarExpr::num_subintervals, VarExprs...>::value : Result;
        };

        template<std::size_t Count, typename VarExpr, typename... VarExprs>
            struct CountDynamicVars<Count, VarExpr, VarExprs...>
        {
            static constexpr unsigned value =
                    VarExpr::is_dynamic ? CountDynamicVars<Count+1, VarExprs...>::value : Count;
        };
    }

    template<typename... VarExprs> struct Variables {
        static_assert(AllOf<IsVariable<VarExprs>::value...>::value,
                      "variables was passed something that is not a variable description!");

        explicit Variables(VarExprs&&... vars) noexcept :
            vars(vars...)
        {}

        using VarTuple = Tuple<VarExprs...>;
        VarTuple vars;

        static constexpr unsigned num_vars = sizeof...(VarExprs);
        static constexpr std::size_t num_dynamic_vars = impl::CountDynamicVars<0, VarExprs...>::value;
        static constexpr std::size_t initial_dynamic_cuboids = impl::ComputeInitialDynamicCuboids<1, VarExprs...>::value;

        template<unsigned VarIndex> using NumSubdivisionsOf = std::integral_constant<unsigned,
                static_cast<unsigned>(TupleElementType<VarIndex, VarTuple>::num_subintervals)>;

        template<std::size_t I> const auto& get() const noexcept {
            return ivarp::get<I>(vars);
        }

        template<std::size_t I>
            const auto& operator[](impl::IndexType<I> ind) const noexcept
        {
            return this->vars[ind];
        }

        using InitialCompileTimeBounds = impl::InitialCompileTimeBounds<VarExprs...>;
    };

    template<typename T> struct IsVariables : std::false_type {};
    template<typename... V> struct IsVariables<Variables<V...>> : std::true_type {};

    template<typename... VarExprs> static inline Variables<BareType<VarExprs>...>
        variables(VarExprs&&... exprs) noexcept
    {
        return Variables<BareType<VarExprs>...>{std::forward<VarExprs>(exprs)...};
    }
}

