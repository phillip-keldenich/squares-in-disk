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
// Created by Phillip Keldenich on 06.12.19.
//

#pragma once

namespace ivarp {
namespace impl {
    template<std::size_t I> struct TypeAtImplI;
    template<> struct TypeAtImplI<0> {
        template<typename A0, typename... Args> struct At {
            using Type = A0;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<1> {
        template<typename A0, typename A1, typename... Args> struct At {
            using Type = A1;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<2> {
        template<typename A0, typename A1, typename A2, typename... Args> struct At {
            using Type = A2;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<3> {
        template<typename A0, typename A1, typename A2, typename A3, typename... Args> struct At {
            using Type = A3;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, A2, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<4> {
        template<typename A0, typename A1, typename A2, typename A3, typename A4, typename... Args> struct At {
            using Type = A4;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename A4, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, A2, A3, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<5> {
        template<typename A0, typename A1, typename A2, typename A3, typename A4,
                 typename A5, typename... Args> struct At {
            using Type = A5;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename A4, typename A5, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, A2, A3, A4, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<6> {
        template<typename A0, typename A1, typename A2, typename A3, typename A4,
                 typename A5, typename A6, typename... Args> struct At {
            using Type = A6;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename A4, typename A5, typename A6, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, A2, A3, A4, A5, NewType, Args...>;
        };
    };
    template<> struct TypeAtImplI<7> {
        template<typename A0, typename A1, typename A2, typename A3, typename A4,
                 typename A5, typename A6, typename A7, typename... Args> struct At {
            using Type = A7;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename... Args>
            struct ReplaceAt
        {
            using Type = InsertInto<A0, A1, A2, A3, A4, A5, A6, NewType, Args...>;
        };
    };
    template<std::size_t I> struct TypeAtImplI {
        template<typename A0, typename A1, typename A2, typename A3, typename A4,
                 typename A5, typename A6, typename A7, typename... Args> struct At
        {
            using Type = typename TypeAtImplI<I-8>::template At<Args...>::Type;
        };
        template<template<typename...> class InsertInto, typename NewType, typename A0, typename A1,
                 typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename... Args>
            struct ReplaceAt
        {
        private:
            template<typename... A> using IntermediateInsertInto = InsertInto<A0,A1,A2,A3,A4,A5,A6,A7,A...>;

        public:
            using Type = typename TypeAtImplI<I-8>::template ReplaceAt<IntermediateInsertInto, NewType, Args...>::Type;
        };
    };
	
	template<template<typename...> class InsertInto, std::size_t I> struct TrailingTypeImplI;
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 0> {
		template<typename A0, typename... Args> struct At {
			using Type = InsertInto<A0,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 1> {
		template<typename A0, typename A1, typename... Args> struct At {
			using Type = InsertInto<A1,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 2> {
		template<typename A0, typename A1, typename A2, typename... Args> struct At {
			using Type = InsertInto<A2,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 3> {
		template<typename A0, typename A1, typename A2, typename A3, typename... Args> struct At {
			using Type = InsertInto<A3,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 4> {
		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename... Args> struct At {
			using Type = InsertInto<A4,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 5> {
		template<typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename... Args> struct At {
			using Type = InsertInto<A5,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 6> {
		template<typename A0, typename A1, typename A2, typename A3, typename A4,
		         typename A5, typename A6, typename... Args> struct At {
			using Type = InsertInto<A6,Args...>;
		};
	};
	template<template<typename...> class InsertInto> struct TrailingTypeImplI<InsertInto, 7> {
		template<typename A0, typename A1, typename A2, typename A3, typename A4,
		         typename A5, typename A6, typename A7, typename... Args> struct At {
			using Type = InsertInto<A7,Args...>;
		};
	};
	template<template<typename...> class InsertInto, std::size_t I> struct TrailingTypeImplI {
		template<typename A0, typename A1, typename A2, typename A3, typename A4,
                 typename A5, typename A6, typename A7, typename... Args> struct At
        {
            using Type = typename TrailingTypeImplI<InsertInto, I-8>::template At<Args...>::Type;
        };
	};

    template<std::size_t I, bool OutOfBounds, typename... Args> struct TypeAtCheckIndex {
        static_assert(!OutOfBounds, "TypeAt index out of bounds!");
    };
    template<std::size_t I, typename... Args> struct TypeAtCheckIndex<I, false, Args...> {
        using Type = typename TypeAtImplI<I>::template At<Args...>::Type;
    };

    template<template<typename...> class InsertInto, std::size_t I, bool OutOfBounds, typename... Args> struct TrailingTypesCheckIndex;
    template<template<typename...> class InsertInto, std::size_t I, typename... Args> struct TrailingTypesCheckIndex<InsertInto, I, true,  Args...> {
        using Type = InsertInto<>;
    };
	template<template<typename...> class InsertInto, std::size_t I, typename... Args> struct TrailingTypesCheckIndex<InsertInto, I, false, Args...> {
		using Type = typename TrailingTypeImplI<InsertInto, I>::template At<Args...>::Type;
	};
}

    template<std::size_t I, typename... Args> using TypeAt =
        typename impl::TypeAtCheckIndex<I, (I >= sizeof...(Args)), Args...>::Type;

    template<template<typename...> class InsertInto, typename NewType, std::size_t I, typename... Args>
        using ReplaceTypeAt = typename impl::TypeAtImplI<I>::template ReplaceAt<InsertInto, NewType, Args...>::Type;
		
	template<template<typename...> class InsertInto, std::size_t I, typename... Args> using TrailingTypes = 
		typename impl::TrailingTypesCheckIndex<InsertInto, I, (I >= sizeof...(Args)), Args...>::Type;
}
