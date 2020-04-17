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
// Created by Phillip Keldenich on 27.11.19.
//

#pragma once

namespace lemma35_proof {
    using namespace ivarp;
    using namespace ivarp::args;

    const auto h = 2_Z*sqrt(maximum(0_Z,1_Z-square(x0)));
    const auto h_defined_bound = (square(x0) <= 1_Z);

    /// Because of the parameters (i, N), we use template structs implementing methods make_fn that  build our actual
    /// functions for concrete values of i and N.
    template<unsigned i, unsigned N> struct b_i {
        static_assert(i >= 1 && i <= N+1, "i out of range!");

        static auto make_fn() {
            return do_make_fn(std::integral_constant<bool, (i == N+1)>{});
        }

    private:
        static auto do_make_fn(std::false_type) {
            return maximum((1_Z/IVARP_CEXPR_CONSTANT(N+1-i)) *
                           (2_Z*T_inv(x0) - IVARP_CEXPR_CONSTANT(i-1)*x0), r(x0));
        }

        static auto do_make_fn(std::true_type) {
            return r(x0);
        }
    };

    template<unsigned i, unsigned N> struct B_i {
        static_assert(i >= 1 && i < N, "i out of range!");

        static auto make_fn() {
            return minimum(x0,
                (1_Z/IVARP_CEXPR_CONSTANT(i)) * (T_inv(x0) + T_inv(r(x0)) - IVARP_CEXPR_CONSTANT(N-i-1) * r(x0)));
        }
    };
    template<unsigned IN> struct B_i<IN,IN> {
        static auto make_fn() {
            return x0;
        }
    };

    template<template<unsigned> class Fn, unsigned First, unsigned LastIncluded> struct TSum {
        using EmptyTag = std::integral_constant<bool, (First > LastIncluded)>;

        static auto make_fn() {
            return do_make_fn(EmptyTag{});
        }

    private:
        // An empty sum is 0.
        template<typename TagType, typename = std::enable_if_t<std::is_same<TagType,std::true_type>::value>>
            static auto do_make_fn(const TagType&, unsigned = 0)
        {
            return 0_Z;
        }

        // A non-empty sum.
        template<typename TagType, typename = std::enable_if_t<std::is_same<TagType,std::false_type>::value>>
            static auto do_make_fn(const TagType&, int = 0 /* make sure this is not a redeclaration */)
        {
            return Fn<First>::make_fn() + TSum<Fn,First+1,LastIncluded>::make_fn();
        }
    };
    // A sum with just one element.
    template<template<unsigned> class Fn, unsigned First> struct TSum<Fn,First,First> {
        static auto make_fn() {
            return Fn<First>::make_fn();
        }
    };

    template<unsigned i, unsigned N> struct S_i {
        static_assert(i >= 1 && i <= N-1, "i out of range!");

        template<unsigned j> using B_j = B_i<j,N>;

        static auto make_fn() {
            return minimum(T_inv(x0) + T_inv(r(x0)) - IVARP_CEXPR_CONSTANT(N-i-1) * r(x0), TSum<B_j, 1u, i>::make_fn());
        }
    };

    template<unsigned i, unsigned N> struct s_i {
        static_assert(i >= 0 && i <= N, "i out of range!");

        template<unsigned j> using b_j = b_i<j,N>;

        static auto make_fn() {
            return TSum<b_j, 1u, i>::make_fn();
        }
    };

    template<unsigned i, unsigned N> struct h_i {
        static_assert(i >= 1 && i < N, "i out of range!");

        static auto make_constr1() {
            return h_defined_bound(T_inv(x0) - s_i<i-1,N>::make_fn());
        }

        static auto make_constr2() {
            return h_defined_bound(T_inv(x0) - S_i<i,N>::make_fn());
        }

        static auto make_fn() {
            return minimum(h(T_inv(x0) - s_i<i-1,N>::make_fn()), h(T_inv(x0) - S_i<i,N>::make_fn()));
        }
    };
    template<unsigned IN> struct h_i<IN,IN> {
        static auto make_fn() {
            return r(x0);
        }
    };

    template<unsigned N> struct S {
        template<unsigned I> struct SumFn {
            static auto make_fn() {
                return maximum((h_i<I,N>::make_fn() - B_i<I,N>::make_fn()) * b_i<I+1,N>::make_fn(),
                               square(b_i<I+1,N>::make_fn()));
            }
        };

        static auto make_fn() {
            return square(x0) + square(b_i<1,N>::make_fn()) + TSum<SumFn, 1u, N>::make_fn();
        }
    };

    using VarSplit = U64Pack<dynamic_subdivision(512, 16)>;

    template<unsigned N> struct System {
        static auto make_system() {
            return do_make_system(IndexRange<1,N>{});
        }

    private:
        template<std::size_t... Inds> static auto do_make_system(IndexPack<Inds...>) {
            const auto s1d = variable(x0, "s_1", 0.295_X, 1.3_X);
            const auto S_fn = S<N>::make_fn();
            return constraint_system(s1d,
                S_fn <= 1.6_X,
                T_inv(x0) + T_inv(r(x0)) > IVARP_CEXPR_CONSTANT(N-1)*r(x0),
                (h_i<unsigned(Inds),N>::make_constr1())...,
                (h_i<unsigned(Inds),N>::make_constr2())...
            );
        }
    };
}
