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
// Created by Phillip Keldenich on 05.10.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Check whether calling type CalledType with argument types ArgsPassed is a symbolic call, i.e.,
    ///  (1) whether the type is symbolically callable, and
    ///  (2) whether at least one argument is a math expression, and
    ///  (3) all arguments are math expressions or implicitly convertible to math constants.
    template<typename CalledType, typename... ArgsPassed> struct IsSymbolicCall :
        std::integral_constant<bool,
            IsMathExprOrPred<CalledType>::value && // (1)
            OneOf<IsMathExpr<ArgsPassed>::value...>::value && // (2)
            AllOf<IsExprOrConstant<ArgsPassed>::value...>::value // (3)
        >
    {};

    /// Prepare the argument tuple for a symbolic call; also check for the right number of arguments.
    template<typename CalledType, typename... ArgsPassed> struct SymbolicPrepareArgs {
        static_assert(NumArgs<CalledType>::value >= sizeof...(ArgsPassed), "Called function with too few arguments!");
        static_assert(NumArgs<CalledType>::value <= sizeof...(ArgsPassed), "Called function with too many arguments!");

        using ArgTuple = Tuple<EnsureExpr<ArgsPassed>...>;

        template<typename... Args> static ArgTuple pack(Args&&... args) {
            return ArgTuple{ensure_expr(ivarp::forward<Args>(args))...};
        }
    };

    /// Metafunction tag for symbolic call evaluation; uses the argument tuple as data.
    struct SymbolicCallMetaFn {};
}
}

namespace ivarp {
    /// Implementation of the SymbolicCallMetaFn metafunction; only args need special care.
    template<typename IndexType> struct MathMetaFn<impl::SymbolicCallMetaFn, MathArg<IndexType>> {
        using OldType = MathArg<IndexType>;
        template<typename ArgTuple> static auto IVARP_H apply(const OldType&, const ArgTuple* data) {
            static constexpr std::size_t I = IndexType::value;
            return ivarp::get<I>(*data);
        }
    };
}

namespace ivarp {
namespace impl {
    template<typename CalledType, typename ArgTuple> static inline IVARP_H auto
        apply_symbolic_call(const CalledType& c, const ArgTuple& args)
    {
        return apply_metafunction<SymbolicCallMetaFn>(c, args);
    }

    template<typename CalledType, typename ArgTuple> struct SymbolicCallResult {
        using Type = decltype(apply_symbolic_call(std::declval<CalledType>(), std::declval<ArgTuple>()));
    };

    template<typename CalledType, typename... Args>
        static inline IVARP_H auto symbolic_call(const CalledType& c, Args&&... args)
    {
        using Prepare = SymbolicPrepareArgs<CalledType, Args...>;
        const typename Prepare::ArgTuple a = Prepare::pack(ivarp::forward<Args>(args)...);
        return apply_symbolic_call(c, a);
    }
}
}

// IMPLEMENTATION OF SYMBOLIC CALL OPERATORS
namespace ivarp {
    template<typename Derived> template<typename... Args, typename V>
        auto MathExpressionBase<Derived>::operator()(Args&&... args) const ->
            impl::SymbolicCallResultType<Derived, typename impl::SymbolicPrepareArgs<Derived, Args...>::ArgTuple>
    {
        return impl::symbolic_call(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
    }

    template<typename Derived> template<typename... Args, typename V>
        auto MathPredBase<Derived>::operator()(Args&&... args) const ->
            impl::SymbolicCallResultType<Derived, typename impl::SymbolicPrepareArgs<Derived, Args...>::ArgTuple>
    {
        return impl::symbolic_call(static_cast<const Derived&>(*this), std::forward<Args>(args)...);
    }
}
