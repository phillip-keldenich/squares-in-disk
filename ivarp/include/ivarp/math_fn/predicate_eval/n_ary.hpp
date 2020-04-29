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
// Created by Phillip Keldenich on 22.11.19.
//

#pragma once

namespace ivarp {
namespace impl {
    /// Implementation of evaluation for n-ary predicates. Here, we need to check
    /// whether the predicate has sequencing semantics (||, &&) or not (|,&).
    template<typename Context, typename NAryPred, typename ArgArray,
             bool IsSequencing = std::is_convertible<typename NAryPred::Tag, MathPredSeq>::value>
        struct NAryPredicateEvaluateImpl;

    /// Implementation of predicate evaluation for non-sequencing predicates (simply forward to tag).
    template<typename Context, typename Tag, typename... Args_, typename ArgArray>
        struct NAryPredicateEvaluateImpl<Context, NAryMathPred<Tag,Args_...>, ArgArray, false>
    {
        using CalledType = NAryMathPred<Tag,Args_...>;
        using ArgTuple = typename CalledType::Args;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline auto eval(const CalledType& c, const ArgArray& args) noexcept(AllowsCUDA<NumberType>::value) {
                return do_eval(c, args, TupleIndexPack<typename CalledType::Args>{});
            }
        )

    private:
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(std::size_t... Indices), NumberType,
            static PredicateEvalResultType<Context> do_eval(const CalledType& c,
                                                            const ArgArray& args,
                                                            IndexPack<Indices...>)
                noexcept(AllowsCUDA<NumberType>::value)
            {
                return invoke_tag<Tag, Context>(args, get<Indices>(c.args)...);
            }
        )
    };

    /// Implementation of predicate evaluation for || and &&.
    template<typename Context, typename Tag, typename... Args_, typename ArgArray>
        struct NAryPredicateEvaluateImpl<Context, NAryMathPred<Tag,Args_...>, ArgArray, true>
    {
        using CalledType = NAryMathPred<Tag,Args_...>;
        using NumberType = typename Context::NumberType;

        IVARP_HD_OVERLOAD_ON_CUDA_NT(NumberType,
            static inline auto eval(const CalledType& c, const ArgArray& args) noexcept(AllowsCUDA<NumberType>::value) {
                return do_eval(c, args, TupleIndexPack<typename CalledType::Args>{});
            }
        )

    private:
        // For ||, early result is true; for &&, early result is false.
        static constexpr bool early_result = std::is_same<Tag, MathPredOrSeq>::value;

        // For combining the truth values, we may use either | or &.
        struct OrCombine {
            IVARP_HD PredicateEvalResultType<Context> operator()(PredicateEvalResultType<Context> a,
                                                        PredicateEvalResultType<Context> b) const noexcept
            {
                return a | b;
            }
        };
        struct AndCombine {
            IVARP_HD PredicateEvalResultType<Context> operator()(PredicateEvalResultType<Context> a,
                                                        PredicateEvalResultType<Context> b) const noexcept
            {
                return a & b;
            }
        };
        using Combine = std::conditional_t<early_result, OrCombine, AndCombine>;

        // Check for early return (for or, definitely true; for and, definitely false).
        template<typename T> static IVARP_HD bool is_early_return(T value, const MathPredOrSeq&) noexcept {
            return definitely(value);
        }
        template<typename T> static IVARP_HD bool is_early_return(T value, const MathPredAndSeq&) noexcept {
            return !possibly(value);
        }

        /// Terminate the recursion: Only one element left.
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(IVARP_TEMPLATE_PARAMS(std::size_t Index1), NumberType,
            static PredicateEvalResultType<Context>
                do_eval(const CalledType& c, const ArgArray& args, IndexPack<Index1>)
                    noexcept(AllowsCUDA<NumberType>::value)
            {
                using ArgTuple = typename CalledType::Args;
                return PredicateEvaluateImpl<Context, TupleElementType<Index1, ArgTuple>, ArgArray>::eval(
                    get<Index1>(c.args), args
                );
            }
        )

        /// At least two elements left: Evaluate the first predicate, check for early termination, and
        /// continue if necessary.
        IVARP_HD_OVERLOAD_TEMPLATE_ON_CUDA_NT(
            IVARP_TEMPLATE_PARAMS(std::size_t Index1, std::size_t Index2, std::size_t... Indices), NumberType,
                static PredicateEvalResultType<Context> do_eval(const CalledType& c, const ArgArray& args,
                                                                IndexPack<Index1,Index2,Indices...>)
                    noexcept(AllowsCUDA<NumberType>::value)
            {
                using ArgTuple = typename CalledType::Args;
                auto r1 = PredicateEvaluateImpl<Context, TupleElementType<Index1, ArgTuple>, ArgArray>::eval(
                    get<Index1>(c.args), args
                );
                if(is_early_return(r1, Tag{})) {
                    return early_result;
                }
                return Combine{}(r1, do_eval(c, args, IndexPack<Index2,Indices...>{}));
            }
        )
    };

    template<typename Context, typename Tag, typename... Args, typename ArgArray>
        struct PredicateEvaluateImpl<Context, NAryMathPred<Tag, Args...>, ArgArray, false> :
            NAryPredicateEvaluateImpl<Context, NAryMathPred<Tag, Args...>, ArgArray>
    {};
}
}
