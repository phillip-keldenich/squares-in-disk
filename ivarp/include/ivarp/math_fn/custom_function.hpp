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

#pragma once

namespace ivarp {
    template<typename FunctorType> struct ContextAsArgumentWrapper {
        template<typename FA, typename = std::enable_if_t<!std::is_same<BareType<FA>, ContextAsArgumentWrapper>::value>>
            explicit ContextAsArgumentWrapper(FA&& functor) :
                functor(std::forward<FA>(functor))
        {}

        ContextAsArgumentWrapper(const ContextAsArgumentWrapper&) = default;
        ContextAsArgumentWrapper(ContextAsArgumentWrapper&&) = default;

        template<typename Context, typename... Args> auto eval(Args&&... args) const {
            return functor(Context{}, args...);
        }

    private:
        FunctorType functor;
    };
    
    template<typename FunctorType, typename... Args, std::enable_if_t<
             AllOf<(IsMathExprOrPred<Args>::value || ImplicitConstantPromotion<Args>::value)...>::value,int> = 0>
        static inline auto custom_function_context_as_template(FunctorType&& functor, Args&&... args)
    {
        return MathCustomFunction<BareType<FunctorType>, EnsureExprOrPred<Args>...>{
            std::forward<FunctorType>(functor), ensure_expr_or_pred(std::forward<Args>(args))...
        };
    }

    template<typename FunctorType, typename... Args, std::enable_if_t<
             AllOf<(IsMathExprOrPred<Args>::value || ImplicitConstantPromotion<Args>::value)...>::value,int> = 0>
        static inline auto custom_function_context_as_arg(FunctorType&& functor, Args&&... args)
    {
        return MathCustomFunction<ContextAsArgumentWrapper<BareType<FunctorType>>, EnsureExprOrPred<Args>...>{
            std::forward<FunctorType>(functor), ensure_expr_or_pred(std::forward<Args>(args))...
        };
    }

    template<typename FunctorType, typename... Args, std::enable_if_t<
             AllOf<(IsMathExprOrPred<Args>::value || ImplicitConstantPromotion<Args>::value)...>::value,int> = 0>
        static inline auto custom_predicate_context_as_template(FunctorType&& functor, Args&&... args)
    {
        return MathCustomPredicate<BareType<FunctorType>, EnsureExprOrPred<Args>...> {
            std::forward<FunctorType>(functor), ensure_expr_or_pred(std::forward<Args>(args))...
        };
    }

    template<typename FunctorType, typename... Args, std::enable_if_t<
             AllOf<(IsMathExprOrPred<Args>::value || ImplicitConstantPromotion<Args>::value)...>::value,int> = 0>
        static inline auto custom_predicate_context_as_arg(FunctorType&& functor, Args&&... args)
    {
        return MathCustomPredicate<ContextAsArgumentWrapper<BareType<FunctorType>>, EnsureExprOrPred<Args>...>{
            std::forward<FunctorType>(functor), ensure_expr_or_pred(std::forward<Args>(args))...
        };
    }
}
