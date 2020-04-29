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
// Created by Phillip Keldenich on 13.12.19.
//

#pragma once

namespace ivarp {
    template<typename MetaFnTag, typename MathExprOrPred> struct MathMetaFn;

    /// Default-implementation: Recursively apply metafunction to children, leaving all other arguments untouched.
    /// The data-parameter is passed through untouched.
    template<typename MetaFnTag, typename IndexType> struct MathMetaFn<MetaFnTag, MathArg<IndexType> > {
        using OldType = MathArg<IndexType>;
        template<typename Data> static IVARP_H OldType apply(const OldType& old, const Data*) noexcept {
            return old;
        }
    };

    template<typename MetaFnTag, typename Tag, typename A1, typename A2>
        struct MathMetaFn<MetaFnTag, MathBinary<Tag,A1,A2> >
    {
        using OldType = MathBinary<Tag,A1,A2>;

        template<typename Data> static IVARP_H auto apply(const OldType& old, const Data* d) {
            auto r1 = MathMetaFn<MetaFnTag, A1>::apply(old.arg1, d);
            auto r2 = MathMetaFn<MetaFnTag, A2>::apply(old.arg2, d);
            using Type = MathBinary<Tag,decltype(r1),decltype(r2)>;
            return Type{ivarp::move(r1), ivarp::move(r2)};
        }
    };

    template<typename MetaFnTag, typename Tag, typename A1, typename A2> struct MathMetaFn<MetaFnTag, BinaryMathPred<Tag,A1,A2> > {
        using OldType = BinaryMathPred<Tag,A1,A2>;

        template<typename Data> static IVARP_H auto apply(const OldType& old, const Data* d) {
            auto r1 = MathMetaFn<MetaFnTag, A1>::apply(old.arg1, d);
            auto r2 = MathMetaFn<MetaFnTag, A2>::apply(old.arg2, d);
            using Type = BinaryMathPred<Tag,decltype(r1),decltype(r2)>;
            return Type{std::move(r1), std::move(r2)};
        }
    };

    template<typename MetaFnTag, typename T, std::int64_t LB, std::int64_t UB>
        struct MathMetaFn<MetaFnTag, MathConstant<T,LB,UB>>
    {
        using OldType = MathConstant<T,LB,UB>;

        template<typename Data>
        static IVARP_H OldType apply(const OldType& t, const Data*) {
            return t;
        }
    };

    template<typename MetaFnTag, std::int64_t LB, std::int64_t UB> struct MathMetaFn<MetaFnTag, MathCUDAConstant<LB,UB>>
    {
        using OldType = MathCUDAConstant<LB,UB>;

        template<typename Data>
        static IVARP_H OldType apply(const OldType& t, const Data*) noexcept {
            return t;
        }
    };

    template<typename MetaFnTag, typename Functor, typename... Args_> struct MathMetaFn<MetaFnTag, MathCustomFunction<Functor, Args_...> > {
        using OldType = MathCustomFunction<Functor, Args_...>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            return do_apply(c, IndexRange<0,sizeof...(Args_)>{}, d);
        }

    private:
        template<typename Data, std::size_t... Inds>
        static IVARP_H auto do_apply(const OldType& c, IndexPack<Inds...>, const Data* d) {
            using TupleType = typename OldType::Args;
            return do_apply_result(
                c.functor,
                (MathMetaFn<MetaFnTag, TupleElementType<Inds, TupleType>>::apply(ivarp::get<Inds>(c.args), d))...
            );
        }

        template<typename F, typename... A_>
            static IVARP_H auto do_apply_result(const F& f, A_&&... args)
        {
            return MathCustomFunction<F, BareType<A_>...>{f, std::move(args)...};
        }
    };

    template<typename MetaFnTag, typename Functor, typename... Args_>
        struct MathMetaFn<MetaFnTag, MathCustomPredicate<Functor, Args_...> >
    {
        using OldType = MathCustomPredicate<Functor, Args_...>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            return do_apply(c, IndexRange<0,sizeof...(Args_)>{}, d);
        }

    private:
        template<typename Data, std::size_t... Inds>
            static IVARP_H auto do_apply(const OldType& c, IndexPack<Inds...>, const Data* d)
        {
            using TupleType = typename OldType::Args;
            return do_apply_result(
                c.functor,
                (MathMetaFn<MetaFnTag, TupleElementType<Inds, TupleType>>::apply(ivarp::get<Inds>(c.args), d))...
            );
        }

        template<typename F, typename... A_>
            static IVARP_H auto do_apply_result(const F& f, A_&&... args)
        {
            return MathCustomPredicate<F, BareType<A_>...>{f, std::move(args)...};
        }
    };

    template<typename MetaFnTag, typename Tag, typename... Args_> struct MathMetaFn<MetaFnTag, MathNAry<Tag, Args_...> > {
        using OldType = MathNAry<Tag, Args_...>;

        template<typename Data>
            static IVARP_H auto apply(const OldType& c, const Data* d)
        {
            return do_apply(c, IndexRange<0,sizeof...(Args_)>{}, d);
        }

    private:
        template<typename Data, std::size_t... Inds>
            static IVARP_H auto do_apply(const OldType& c, IndexPack<Inds...>, const Data* d)
        {
            using TupleType = typename OldType::Args;
            return do_apply_result(
                (MathMetaFn<MetaFnTag, TupleElementType<Inds, TupleType>>::apply(ivarp::get<Inds>(c.args), d))...);
        }

        template<typename... A_>
            static IVARP_H auto do_apply_result(A_&&... args)
        {
            return MathNAry<Tag, BareType<A_>...>{std::move(args)...};
        }
    };

    template<typename MetaFnTag, typename Tag, typename... Args_> struct MathMetaFn<MetaFnTag, NAryMathPred<Tag, Args_...> > {
        using OldType = NAryMathPred<Tag, Args_...>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            return do_apply(c, IndexRange<0,sizeof...(Args_)>{}, d);
        }

    private:
        template<typename Data, std::size_t... Inds>
            static IVARP_H auto do_apply(const OldType& c, IndexPack<Inds...>, const Data* d)
        {
            using TupleType = typename OldType::Args;
            return do_apply_result(
                (MathMetaFn<MetaFnTag, TupleElementType<Inds, TupleType>>::apply(ivarp::get<Inds>(c.args), d))...
            );
        }

        template<typename... A_>
            static IVARP_H auto do_apply_result(A_&&... args)
        {
            return NAryMathPred<Tag, BareType<A_>...>{std::move(args)...};
        }
    };

    template<typename MetaFnTag, typename Tag, typename A1, typename A2, typename A3> struct MathMetaFn<MetaFnTag, MathTernary<Tag,A1,A2,A3>> {
        using OldType = MathTernary<Tag,A1,A2,A3>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            auto r1 = MathMetaFn<MetaFnTag, A1>::apply(c.arg1, d);
            auto r2 = MathMetaFn<MetaFnTag, A2>::apply(c.arg2, d);
            auto r3 = MathMetaFn<MetaFnTag, A3>::apply(c.arg3, d);
            using Type = MathTernary<Tag,decltype(r1),decltype(r2),decltype(r3)>;
            return Type{std::move(r1), std::move(r2), std::move(r3)};
        }
    };

    template<typename MetaFnTag, typename Tag, typename A1> struct MathMetaFn<MetaFnTag, MathUnary<Tag,A1>> {
        using OldType = MathUnary<Tag,A1>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            auto r1 = MathMetaFn<MetaFnTag, A1>::apply(c.arg, d);
            using Type = MathUnary<Tag,decltype(r1)>;
            return Type{std::move(r1)};
        }
    };

    template<typename MetaFnTag, typename Tag, typename A1> struct MathMetaFn<MetaFnTag, UnaryMathPred<Tag,A1>> {
        using OldType = UnaryMathPred<Tag,A1>;

        template<typename Data>
        static IVARP_H auto apply(const OldType& c, const Data* d) {
            auto r1 = MathMetaFn<MetaFnTag, A1>::apply(c.arg, d);
            using Type = UnaryMathPred<Tag,decltype(r1)>;
            return Type{std::move(r1)};
        }
    };

    template<typename MetaFnTag, typename T, bool LB, bool UB> struct MathMetaFn<MetaFnTag, MathBoolConstant<T,LB,UB>> {
        using OldType = MathBoolConstant<T,LB,UB>;

        template<typename Data>
        static IVARP_H OldType apply(const OldType& t, const Data*) noexcept {
            return t;
        }
    };

    template<typename MetaFnTag, typename MathExpr, typename BoundsType>
        struct MathMetaFn<MetaFnTag, BoundedMathExpr<MathExpr, BoundsType>>
    {
        using OldType = BoundedMathExpr<MathExpr, BoundsType>;

        /// Default implementation drops bounds; we cannot know what the transformation does.
        template<typename Data>
        static IVARP_H auto apply(const OldType& t, const Data* d) {
            return MathMetaFn<MetaFnTag, MathExpr>::apply(t.child, d);
        }
    };

    template<typename MetaFnTag, typename MathPred, typename BoundsType>
        struct MathMetaFn<MetaFnTag, BoundedPredicate<MathPred, BoundsType>>
    {
        using OldType = BoundedPredicate<MathPred, BoundsType>;

        /// Default implementation drops bounds; we cannot know what the transformation does.
        template<typename Data>
        static IVARP_H auto apply(const OldType& t, const Data* d) {
            return MathMetaFn<MetaFnTag, MathPred>::apply(t.child, d);
        }
    };

    template<typename MetaFnTag, typename Child, std::int64_t LB, std::int64_t UB>
        struct MathMetaFn<MetaFnTag, ConstantFoldedExpr<Child, LB, UB>>
    {
        using OldType = ConstantFoldedExpr<Child, LB, UB>;

        /// Default implementation drops bounds/constant folding; we cannot know what the transformation does.
        template<typename Data>
        static IVARP_H auto apply(const OldType& t, const Data* d) {
            return MathMetaFn<MetaFnTag, Child>::apply(t.base, d);
        }
    };

    template<typename MetaFnTag, typename Child, bool LB, bool UB>
        struct MathMetaFn<MetaFnTag, ConstantFoldedPred<Child, LB, UB>>
    {
        using OldType = ConstantFoldedPred<Child, LB, UB>;

        /// Default implementation drops bounds/constant folding; we cannot know what the transformation does.
        template<typename Data>
        static IVARP_H auto apply(const OldType& t, const Data* d) {
            return MathMetaFn<MetaFnTag, Child>::apply(t.base, d);
        }
    };

    /// Apply metafunction without data, passing a const void nullptr as data.
    template<typename MetaTag, typename MathExprOrPred>
        static inline IVARP_H auto apply_metafunction(const MathExprOrPred& input)
    {
        return MathMetaFn<MetaTag, MathExprOrPred>::apply(input, static_cast<const void*>(nullptr));
    }

    /// Apply metafunction with data, passing a pointer to the given data.
    template<typename MetaTag, typename MathExprOrPred, typename Data>
        static inline IVARP_H auto apply_metafunction(const MathExprOrPred& input, const Data& data)
    {
        return MathMetaFn<MetaTag, MathExprOrPred>::apply(input, &data);
    }
}
