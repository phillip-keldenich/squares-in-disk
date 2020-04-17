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
// Created by Phillip Keldenich on 06.02.20.
//

#pragma once

namespace ivarp {
namespace impl {
    template<typename MAT, typename LB, typename UB> static inline IVARP_H auto
        extract_constraints_from_description(const VariableDescription<MAT,LB,UB>& d)
    {
        return ivarp::make_tuple((MAT{} >= d.lb), (MAT{} <= d.ub));
    }

    template<typename MAT, typename C> static inline IVARP_H auto
        extract_constraints_from_description(const ValueDescription<MAT,C>& d)
    {
        return ivarp::make_tuple(MAT{} == d.expr);
    }

    template<typename ArgDescriptionTuple, std::size_t... Indices>
        static inline IVARP_H auto extract_bound_constraints(const ArgDescriptionTuple& t, IndexPack<Indices...>)
    {
        return concat_tuples(extract_constraints_from_description(ivarp::template get<Indices>(t))...);
    }

    template<typename ArgDescriptionTuple>
        static inline IVARP_H auto extract_bound_constraints(const ArgDescriptionTuple& t)
    {
        return extract_bound_constraints(t, TupleIndexPack<ArgDescriptionTuple>{});
    }
}
}
