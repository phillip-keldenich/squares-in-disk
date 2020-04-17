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
// Created by Phillip Keldenich on 24.10.19.
//

#pragma once

/**
 * @file metaprogramming.hpp
 * Template metaprogramming helpers.
 */

#include <tuple>
#include <utility>
#include <type_traits>

#include "ivarp/cuda.hpp"
#include "metaprogramming/bare_type.hpp"
#include "metaprogramming/allof_oneof.hpp"
#include "metaprogramming/minof_maxof.hpp"
#include "metaprogramming/index_pack.hpp"
#include "metaprogramming/index_at.hpp"
#include "metaprogramming/index_concat.hpp"
#include "metaprogramming/index_add_offset.hpp"
#include "metaprogramming/index_range.hpp"
#include "metaprogramming/merge_index_packs.hpp"
#include "metaprogramming/filter_index_pack.hpp"
#include "metaprogramming/tuple_index_pack.hpp"
#include "metaprogramming/type_at.hpp"
#include "metaprogramming/predicate_not.hpp"

namespace ivarp {
    /// Used to wrap parameter packs.
    template<typename... Args> struct ArgPack {
        template<typename T> struct Prepend {
            using Type = ArgPack<T, Args...>;
        };
        template<typename T> struct Append {
            using Type = ArgPack<Args..., T>;
        };
    };

    template<typename... T> struct MakeVoidImpl {
        using Type = void;
    };
    template<typename... T> using MakeVoid = typename MakeVoidImpl<T...>::Type;

    template<typename T, template<T...> class InsertInto, T... Sequence> struct ReverseImpl {
        using Type = InsertInto<>;
    };

    template<typename T, template<T...> class InsertInto, T S1, T... Rest> struct ReverseImpl<T, InsertInto, S1, Rest...> {
        template<T... Seq> using BackInsert = InsertInto<Seq..., S1>;
        using Type = typename ReverseImpl<T, BackInsert, Rest...>::Type;
    };

    template<typename T, template<T...> class InsertInto, T... Sequence> using Reverse = typename ReverseImpl<T,InsertInto,Sequence...>::Type;

    namespace impl {
        template<std::size_t I> struct IndexType {
            static constexpr std::size_t value = I;

            template<typename... Ts> IVARP_H auto& operator[](std::tuple<Ts...>& t) const noexcept {
                return std::get<value>(t);
            }
            template<typename... Ts> IVARP_H const auto& operator[](const std::tuple<Ts...>& t) const noexcept {
                return std::get<value>(t);
            }
            template<typename... Ts> IVARP_H auto&& operator[](std::tuple<Ts...>&& t) const noexcept {
                return std::get<value>(std::move(t));
            }

            template<typename... Ts> inline IVARP_HD auto& operator[](Tuple<Ts...>& t) const noexcept;
            template<typename... Ts> inline IVARP_HD const auto& operator[](const Tuple<Ts...>& t) const noexcept;
            template<typename... Ts> inline IVARP_HD auto&& operator[](Tuple<Ts...>&& t) const noexcept;
        };

        template<char... Digits> struct ValueOf;
        template<char Digit> struct ValueOf<Digit> {
            static_assert(Digit >= '0' && Digit <= '9', "Wrong digit!");

            static constexpr std::size_t multiplier = 1;
            static constexpr std::size_t value = Digit - '0';
        };
        template<char D1, char D2, char... Digits> struct ValueOf<D1,D2,Digits...> {
            static_assert(D1 >= '0' && D1 <= '9', "Wrong digit!");

            using NextType = ValueOf<D2,Digits...>;
            static constexpr std::size_t multiplier = 10 * NextType::multiplier;
            static constexpr std::size_t value = multiplier * (D1 - '0') + NextType::value;
        };
    }

    template<char... Digits> IVARP_HD constexpr auto operator""_i() noexcept { return impl::IndexType<impl::ValueOf<Digits...>::value>{}; }

#define IVARP_IND(integral_constexpr) (::ivarp::impl::IndexType<(integral_constexpr)>{})


    template<template<typename...> class InsertInto, typename ArgsPack> struct InsertArgs;
    template<template<typename...> class InsertInto, typename... Args> struct InsertArgs<InsertInto, ArgPack<Args...>> {
        using Type = InsertInto<Args...>;
    };

    /// Remove all elements from UnfilteredPack that correspond to a false entry in Predicate.
    template<typename UnfilteredPack, bool... Predicate> struct FilterPack;
    template<> struct FilterPack<ArgPack<>> {
        using Type = ArgPack<>;
    };
    template<typename A, typename... Args, bool... Predicate>
        struct FilterPack<ArgPack<A, Args...>, false, Predicate...>
    {
        using Type = typename FilterPack<ArgPack<Args...>, Predicate...>::Type;
    };
    template<typename A, typename... Args, bool... Predicate>
        struct FilterPack<ArgPack<A, Args...>, true, Predicate...>
    {
    private:
        using BaseType = typename FilterPack<ArgPack<Args...>, Predicate...>::Type;

    public:
        using Type = typename BaseType::template Prepend<A>::Type;
    };

    /// Filter the argument tuple Args according to PredicateType<Args>::value..., and insert the types
    /// for which the predicate is true into InsertInto.
    template<template<typename...> class InsertInto, template<typename> class PredicateType, typename... Args>
        struct FilterArgs
    {
    private:
        using UnfPack = ArgPack<Args...>;
        using FPack = typename FilterPack<UnfPack, PredicateType<Args>::value...>::Type;

    public:
        using Type = typename InsertArgs<InsertInto, FPack>::Type;
    };
    template<template<typename...> class InsertInto, template<typename> class PredicateType, typename... Args>
        using FilterArgsType = typename FilterArgs<InsertInto,PredicateType,Args...>::Type;

    /// For a sequence of n types, generate an IndexPack containing indices for which Predicate returns true.
    template<template<typename> class Predicate, std::size_t Current, typename... Args>
    struct FilteredIndexPack {
        using Type = IndexPack<>;
    };
    template<template<typename> class Predicate, std::size_t Current, bool CurrentTrue, typename... Args>
    struct FilteredIndexPackImpl {
        using Type = typename FilteredIndexPack<Predicate, Current+1, Args...>::Type;
    };
    template<template<typename> class Predicate, std::size_t Current, typename... Args>
    struct FilteredIndexPackImpl<Predicate, Current, true, Args...> {
        using Type = typename FilteredIndexPack<Predicate, Current+1, Args...>::Type::template Prepend<Current>::Type;
    };
    template<template<typename> class Predicate, std::size_t Current, typename A1, typename... Args>
        struct FilteredIndexPack<Predicate, Current, A1, Args...> :
            FilteredIndexPackImpl<Predicate, Current, Predicate<A1>::value, Args...>
    {};
    template<template<typename> class Predicate, typename... Args> using FilteredIndexPackType =
        typename FilteredIndexPack<Predicate, 0, Args...>::Type;

    /// Check if T occurs in the parameter pack TList.
    template<typename T, typename... TList> struct TypeIn : OneOf<std::is_same<T,TList>::value...> {};

    /// Put each value in the given IndexPack into the given template
    /// and put the resulting sequence of types into InsertInto.
    template<template<typename...> class InsertInto, template<std::size_t> class Template, typename IndexPack>
        struct ForEachIndex;
    template<template<typename...> class InsertInto, template<std::size_t> class Template, std::size_t... Inds>
        struct ForEachIndex<InsertInto, Template, IndexPack<Inds...>>
    {
        using Type = InsertInto<Template<Inds>...>;
    };
    template<template<typename...> class InsertInto, template<std::size_t> class Template, typename IndexPack>
        using ForEachIndexType = typename ForEachIndex<InsertInto, Template, IndexPack>::Type;

    static_assert(std::is_same<decltype(12345678_i), decltype(IVARP_IND(12345678ul))>::value, "Error!");
}
