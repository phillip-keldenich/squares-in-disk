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
// Created by Phillip Keldenich on 05.12.19.
//

#pragma once

#include "metaprogramming.hpp"
#include "cuda.hpp"

namespace ivarp {
namespace impl {
	template<std::size_t I, typename... Args> struct TupleContentImpl;
    template<std::size_t I> struct TupleContentImpl<I> { static constexpr std::size_t size = I; };
	
    template<typename TupleType, std::size_t I> struct TupleElementImpl;

    /// For the first elements, optimize the type lookup (w.r.t. compile times).
#define IVARP_TUPLE_ENTRY_SPECIAL(index)\
	template<typename T, typename... Args> struct TupleContentImpl<index, T, Args...> : \
		TupleContentImpl<index+1, Args...>\
	{\
	private:\
		template<typename A> using DisableSame = std::enable_if_t<!std::is_same<BareType<A>, TupleContentImpl>::value>;\
	public:\
		TupleContentImpl() = default;\
        IVARP_DEFAULT_CM(TupleContentImpl);\
		IVARP_SUPPRESS_HD\
		template<typename A, typename... AA, typename = DisableSame<A>> IVARP_HD TupleContentImpl(A&& a, AA&&... args) \
			noexcept(noexcept(TupleContentImpl<index+1, Args...>(ivarp::forward<AA>(args)...)) && noexcept(T(ivarp::forward<A>(a)))) :\
				TupleContentImpl<index+1, Args...>(ivarp::forward<AA>(args)...), m_value(ivarp::forward<A>(a)) {}\
		using EntryT##index = T;\
		using TupleEntryT##index = TupleContentImpl<index, T, Args...>;\
		IVARP_HD T& get() noexcept { return m_value; }\
		IVARP_HD const T& get() const noexcept { return m_value; }\
		IVARP_HD T& get ## index () noexcept { return m_value; }\
        IVARP_HD const T& get ## index () const noexcept { return m_value; }\
		T m_value;\
	};\
    template<typename TupleType> struct TupleElementImpl<TupleType, index> {\
        static_assert(TupleType::size > index, "Index out of range!");\
		using EntryType = typename TupleType::TupleEntryT ## index ;\
        using Type = typename TupleType::EntryT ## index ;\
        static IVARP_HD Type &get(TupleType& t) noexcept {\
			return t.get ## index ();\
        }\
		static IVARP_HD const Type &get(const TupleType& t) noexcept {\
			return t.get ## index ();\
		}\
    }

    IVARP_TUPLE_ENTRY_SPECIAL( 0);
    IVARP_TUPLE_ENTRY_SPECIAL( 1);
    IVARP_TUPLE_ENTRY_SPECIAL( 2);
    IVARP_TUPLE_ENTRY_SPECIAL( 3);
    IVARP_TUPLE_ENTRY_SPECIAL( 4);
    IVARP_TUPLE_ENTRY_SPECIAL( 5);
    IVARP_TUPLE_ENTRY_SPECIAL( 6);
    IVARP_TUPLE_ENTRY_SPECIAL( 7);
    IVARP_TUPLE_ENTRY_SPECIAL( 8);
    IVARP_TUPLE_ENTRY_SPECIAL( 9);
    IVARP_TUPLE_ENTRY_SPECIAL(10);
    IVARP_TUPLE_ENTRY_SPECIAL(11);
    IVARP_TUPLE_ENTRY_SPECIAL(12);
    IVARP_TUPLE_ENTRY_SPECIAL(13);
    IVARP_TUPLE_ENTRY_SPECIAL(14);
    IVARP_TUPLE_ENTRY_SPECIAL(15);
#undef IVARP_TUPLE_ENTRY_SPECIAL

	template<std::size_t I, typename T, typename... Args> struct TupleContentImpl<I, T, Args...> :
		TupleContentImpl<I+1,Args...>
	{
	private:
        static_assert(I >= 16, "Wrong specialization chosen!");
		template<typename A> using DisableSame = std::enable_if_t<!std::is_same<BareType<A>, TupleContentImpl>::value>;
		
	public:
        using Type = T;

		TupleContentImpl() = default;
        IVARP_DEFAULT_CM(TupleContentImpl);

		template<typename A, typename... AA, typename = DisableSame<A>> IVARP_HD TupleContentImpl(A&& a, AA&&... args)
			noexcept(noexcept(TupleContentImpl<I+1, Args...>(ivarp::forward<AA>(args)...), T(ivarp::forward<A>(a)))) :
				TupleContentImpl<I+1, Args...>(ivarp::forward<AA>(args)...), m_value(ivarp::forward<A>(a))
		{}
		
		IVARP_HD T& get() noexcept { return m_value; }
		IVARP_HD const T& get() const noexcept { return m_value; }
		
		T m_value;
	};

	template<typename TT, std::size_t I> struct HighIndexTupleElementImpl;
	template<typename... Args, std::size_t I> struct HighIndexTupleElementImpl<Tuple<Args...>, I> {
        template<typename... Args_> using EntryTypeNoArgs = TupleContentImpl<I, Args_...>;
        using EntryType = TrailingTypes<EntryTypeNoArgs, I, Args...>;
        using Type = typename EntryType::Type;

		static IVARP_HD Type &get(Tuple<Args...>& t) noexcept {
			return static_cast<EntryType&>(t).get();
		}
		static IVARP_HD const Type& get(const Tuple<Args...>& t) noexcept {
			return static_cast<const EntryType&>(t).get();
		}
	};
    template<typename TupleType, std::size_t I> struct TupleElementImpl :
        HighIndexTupleElementImpl<TupleType, I>
    {};
}
}

namespace ivarp {
    /// A device-compatible minimal tuple type (not completely compatible with std::tuple;
    /// don't put const/volatile types or references in it).
    template<typename... Types> struct Tuple : impl::TupleContentImpl<0u, Types...> {
        using Base = impl::TupleContentImpl<0u, Types...>;

        /// Default/copy/move constructors are implicit.
        Tuple() = default;
        IVARP_DEFAULT_CM(Tuple);

        template<typename A1, typename=std::enable_if_t<!std::is_same<BareType<A1>, Tuple>::value>>
            IVARP_HD explicit Tuple(A1&& a1) noexcept(noexcept(Base(ivarp::forward<A1>(a1)))) :
                Base{ivarp::forward<A1>(a1)}
        {}

        template<typename A1, typename A2, typename... Args>
            IVARP_HD Tuple(A1&& a1, A2&& a2, Args&&... args) noexcept(noexcept(
                Base{ivarp::forward<A1>(a1), ivarp::forward<A2>(a2), ivarp::forward<Args>(args)...})) :
                Base{ivarp::forward<A1>(a1), ivarp::forward<A2>(a2), ivarp::forward<Args>(args)...}
        {}

        template<std::size_t I> using At = typename impl::TupleElementImpl<Tuple, I>::Type;

        template<typename T> using Append = Tuple<Types..., T>;
        template<std::size_t I, typename T> using ReplaceAt = ReplaceTypeAt<Tuple, T, I, Types...>;

        template<std::size_t I> inline IVARP_HD auto& operator[](impl::IndexType<I> i) noexcept {
            return i[*this];
        }

        template<std::size_t I> inline IVARP_HD const auto& operator[](impl::IndexType<I> i) const noexcept {
            return i[*this];
        }
    };

	namespace impl {
		template<std::size_t Index, typename TupleType> struct TupleElementTypeImpl;
		template<std::size_t Index, typename... Args> struct TupleElementTypeImpl<Index, Tuple<Args...>> {
			static_assert(Index < sizeof...(Args), "TupleElementType index out of range!");
			using Type = typename impl::TupleElementImpl<Tuple<Args...>, Index>::Type;
		};
		template<typename TupleType> struct TupleSizeImpl;
		template<typename... Args> struct TupleSizeImpl<Tuple<Args...>> {
			using Type = std::integral_constant<std::size_t, sizeof...(Args)>;
            static_assert(sizeof...(Args) == Tuple<Args...>::size, "Wrong size!");
		};
	}
	template<std::size_t Index, typename TupleType> using TupleElementType = typename impl::TupleElementTypeImpl<Index, TupleType>::Type;
	template<typename TupleType> using TupleSize = typename impl::TupleSizeImpl<TupleType>::Type;

    /// std::get replacement.
    template<std::size_t I, typename... Types> static inline IVARP_HD
        auto& get(Tuple<Types...>& t) noexcept
    {
        static_assert(I < sizeof...(Types), "get: Index out of range!");
        return impl::template TupleElementImpl<Tuple<Types...>, I>::get(t);
    }
    template<std::size_t I, typename... Types> static inline IVARP_HD
        const auto& get(const Tuple<Types...>& t) noexcept
    {
        static_assert(I < sizeof...(Types), "get: Index out of range!");
        return impl::template TupleElementImpl<Tuple<Types...>, I>::get(t);
    }
    template<std::size_t I, typename... Types> static inline IVARP_HD
        auto&& get(Tuple<Types...>&& t)
    {
        static_assert(I < sizeof...(Types), "get: Index out of range!");
        return ivarp::move(impl::template TupleElementImpl<Tuple<Types...>, I>::get(t));
    }
    template<std::size_t I, typename... Types> static inline IVARP_HD
        const auto&& get(const Tuple<Types...>&& t)
    {
        static_assert(I < sizeof...(Types), "get: Index out of range!");
        return ivarp::move(impl::template TupleElementImpl<Tuple<Types...>, I>::get(t));
    }

    template<typename... Types> static inline IVARP_H auto make_tuple(Types&&... types) {
        return Tuple<BareType<Types>...>(ivarp::forward<Types>(types)...);
    }

    template<typename TupleType, std::size_t... Indices>
    static inline IVARP_H auto filter_tuple(TupleType&& t, IndexPack<Indices...> /*selected_indices*/)
    {
        return ivarp::make_tuple(ivarp::template get<Indices>(ivarp::forward<TupleType>(t))...);
    }

    template<typename... Types> static inline IVARP_HD auto make_device_compatible_tuple(Types&&... types) noexcept {
        return Tuple<BareType<Types>...>(ivarp::forward<Types>(types)...);
    }

    namespace prepend_tuple_impl {
        template<typename... Types, std::size_t... Indices, typename... PrependArgs> static inline IVARP_H
            auto do_prepend_tuple(const Tuple<Types...>& p, IndexPack<Indices...>, PrependArgs&&... prepend)
        {
            return Tuple<BareType<PrependArgs>..., Types...>{
				ivarp::forward<PrependArgs>(prepend)...,
                ivarp::template get<Indices>(p)...
            };
        }

        template<typename... Types, std::size_t... Indices, typename... PrependArgs> static inline IVARP_H
            auto do_prepend_tuple(Tuple<Types...>&& p, IndexPack<Indices...>, PrependArgs&&... prepend)
        {
            return Tuple<BareType<PrependArgs>..., Types...>{
				ivarp::forward<PrependArgs>(prepend)...,
                ivarp::template get<Indices>(ivarp::move(p))...
            };
        }
    }

    /// Prepend the tuple given as first parameter by the remaining parameters.
    template<typename... Types, typename... PrependArgs>
        static inline IVARP_H auto prepend_tuple(const Tuple<Types...>& p, PrependArgs&&... prepend)
    {
        using Range = IndexRange<0, sizeof...(Types)>;
        return prepend_tuple_impl::do_prepend_tuple(p, Range{}, ivarp::forward<PrependArgs>(prepend)...);
    }
    template<typename... Types, typename... PrependArgs>
        static inline IVARP_H auto prepend_tuple(Tuple<Types...>&& p, PrependArgs&&... prepend)
    {
        using Range = IndexRange<0, sizeof...(Types)>;
        return prepend_tuple_impl::do_prepend_tuple(ivarp::move(p), Range{}, ivarp::forward<PrependArgs>(prepend)...);
    }

    /// Concatenate the given tuples.
    static inline IVARP_H Tuple<> concat_tuples() {
        return Tuple<>{};
    }

    template<typename T1>
        static inline IVARP_H auto concat_tuples(T1&& t1)
    {
        return ivarp::forward<T1>(t1);
    }

    template<typename T1, typename T2, std::size_t... I1, std::size_t... I2>
        static inline IVARP_H auto concat_tuple_impl(T1&& t1, T2&& t2, IndexPack<I1...>, IndexPack<I2...>)
    {
        return ivarp::make_tuple(ivarp::template get<I1>(ivarp::forward<T1>(t1))...,
                                 ivarp::template get<I2>(ivarp::forward<T2>(t2))...);
    }

    template<typename T1, typename T2, typename... TRest>
        static inline IVARP_H auto concat_tuples(T1&& t1, T2&& t2, TRest&&... rest)
    {
        return concat_tuples(concat_tuple_impl(ivarp::forward<T1>(t1), ivarp::forward<T2>(t2),
                                               TupleIndexPack<T1>{}, TupleIndexPack<T2>{}),
                             ivarp::forward<TRest>(rest)...);
    }

    enum class TupleVisitationControl {
        CONTINUE,
        STOP
    };

    namespace impl {
        template<typename Visitor, typename T,
                 std::enable_if_t<std::is_void<decltype(std::declval<Visitor>()(std::declval<T>()))>::value,int> = 0>
            static inline IVARP_H TupleVisitationControl visit_tuple_invoke_visitor(Visitor&& v, T&& t)
        {
            (ivarp::forward<Visitor>(v))(ivarp::forward<T>(t));
            return TupleVisitationControl::CONTINUE;
        }

        template<typename Visitor, typename T,
                 std::enable_if_t<!std::is_void<decltype(std::declval<Visitor>()(std::declval<T>()))>::value,int> = 0>
            static inline IVARP_H TupleVisitationControl visit_tuple_invoke_visitor(Visitor&& v, T&& t)
        {
            return (ivarp::forward<Visitor>(v))(ivarp::forward<T>(t));
        }

        template<typename Visitor, typename TupleType, std::size_t I1, std::size_t... Inds>
            static inline IVARP_H void visit_tuple_impl(Visitor&& v, TupleType&& t, IndexPack<I1,Inds...>)
        {
            auto&& a = ivarp::template get<I1>(ivarp::forward<TupleType>(t));
            auto vc = visit_tuple_invoke_visitor(ivarp::forward<Visitor>(v), ivarp::forward<decltype(a)>(a));
            if(vc == TupleVisitationControl::STOP) {
                return;
            }
            visit_tuple_impl(ivarp::forward<Visitor>(v), ivarp::forward<TupleType>(t), IndexPack<Inds...>{});
        }

        template<typename Visitor, typename TupleType>
            static inline IVARP_H void visit_tuple_impl(Visitor&& v, TupleType&& t, IndexPack<>)
        {}
    }

    template<typename Visitor, typename... Types> static inline IVARP_H
        void visit_tuple(Visitor&& v, Tuple<Types...>& t)
    {
        impl::visit_tuple_impl(ivarp::forward<Visitor>(v), t, IndexRange<0,sizeof...(Types)>{});
    }

    template<typename Visitor, typename... Types> static inline IVARP_H
        void visit_tuple(Visitor&& v, Tuple<Types...>&& t)
    {
        impl::visit_tuple_impl(ivarp::forward<Visitor>(v), ivarp::move(t), IndexRange<0,sizeof...(Types)>{});
    }

    template<typename Visitor, typename... Types> static inline IVARP_H
        void visit_tuple(Visitor&& v, const Tuple<Types...>& t)
    {
        impl::visit_tuple_impl(ivarp::forward<Visitor>(v), t, IndexRange<0,sizeof...(Types)>{});
    }

    namespace impl {
        template<typename Visitor, typename T,
                 std::enable_if_t<std::is_void<decltype(std::declval<Visitor>()(std::declval<T>()))>::value,int> = 0>
            static inline IVARP_HD TupleVisitationControl visit_tuple_hd_invoke_visitor(Visitor&& v, T&& t)
        {
            (ivarp::forward<Visitor>(v))(ivarp::forward<T>(t));
            return TupleVisitationControl::CONTINUE;
        }

        template<typename Visitor, typename T,
                 std::enable_if_t<!std::is_void<decltype(std::declval<Visitor>()(std::declval<T>()))>::value,int> = 0>
            static inline IVARP_HD TupleVisitationControl visit_tuple_hd_invoke_visitor(Visitor&& v, T&& t)
        {
            return (ivarp::forward<Visitor>(v))(ivarp::forward<T>(t));
        }

        template<typename Visitor, typename TupleType, std::size_t I1, std::size_t... Inds>
            static inline IVARP_HD void visit_tuple_hd_impl(Visitor&& v, TupleType&& t, IndexPack<I1,Inds...>)
        {
            auto&& a = ivarp::template get<I1>(ivarp::forward<TupleType>(t));
            auto vc = visit_tuple_hd_invoke_visitor(ivarp::forward<Visitor>(v), ivarp::forward<decltype(a)>(a));
            if(vc == TupleVisitationControl::STOP) {
                return;
            }
            visit_tuple_hd_impl(ivarp::forward<Visitor>(v), ivarp::forward<TupleType>(t), IndexPack<Inds...>{});
        }

        template<typename Visitor, typename TupleType>
            static inline IVARP_HD void visit_tuple_hd_impl(Visitor&& v, TupleType&& t, IndexPack<>)
        {}
    }

    template<typename Visitor, typename... Types> static inline IVARP_HD
        void visit_tuple_hd(Visitor&& v, Tuple<Types...>& t)
    {
        impl::visit_tuple_hd_impl(ivarp::forward<Visitor>(v), t, IndexRange<0,sizeof...(Types)>{});
    }

    template<typename Visitor, typename... Types> static inline IVARP_HD
        void visit_tuple_hd(Visitor&& v, Tuple<Types...>&& t)
    {
        impl::visit_tuple_hd_impl(ivarp::forward<Visitor>(v), ivarp::move(t), IndexRange<0,sizeof...(Types)>{});
    }

    template<typename Visitor, typename... Types> static inline IVARP_HD
        void visit_tuple_hd(Visitor&& v, const Tuple<Types...>& t)
    {
        impl::visit_tuple_hd_impl(ivarp::forward<Visitor>(v), t, IndexRange<0,sizeof...(Types)>{});
    }
}

template<std::size_t I> template<typename... Ts> auto& ivarp::impl::IndexType<I>::operator[](Tuple<Ts...>& t) const noexcept {
	return ivarp::get<I>(t);
}

template<std::size_t I> template<typename... Ts> const auto& ivarp::impl::IndexType<I>::operator[](const Tuple<Ts...>& t) const noexcept {
    return ivarp::get<I>(t);
}

template<std::size_t I> template<typename... Ts> auto&& ivarp::impl::IndexType<I>::operator[](Tuple<Ts...>&& t) const noexcept {
	return ivarp::get<I>(ivarp::move(t));
}
