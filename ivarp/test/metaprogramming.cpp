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
#include "ivarp/metaprogramming.hpp"
#include "ivarp/tuple.hpp"
#include "ivarp/math_fn.hpp"
#include <limits>
#include <tuple>

using namespace ivarp;

namespace test_allof_oneof {
    static_assert(!AllOf<true, true, false, true>::value, "Error!");
    static_assert(OneOf<true, true, false, true>::value, "Error!");
    static_assert(AllOf<>::value, "Error!");
    static_assert(!OneOf<>::value, "Error!");
    static_assert(!AllOf<std::is_integral<double>::value>::value, "Error!");
    static_assert(!OneOf<std::is_integral<double>::value>::value, "Error!");
}

namespace test_minof_maxof {
    static_assert(MinOf<>::value == std::numeric_limits<std::size_t>::max(), "Error!");
    static_assert(MaxOf<>::value == 0, "Error!");
    static_assert(MinOf<1, 1, 9, 0, 55, 1, 2>::value == 0, "Error!");
    static_assert(MaxOf<1, 1, 9, 0, 55, 1, 2>::value == 55, "Error!");
}

namespace test_index_range {
    static_assert(std::is_same<IndexRange<0, 0>, IndexPack<>>::value, "Error!");
    static_assert(std::is_same<IndexRange<100, 100>, IndexPack<>>::value, "Error!");
    static_assert(std::is_same<IndexRange<1001, 1000>, IndexPack<>>::value, "Error!");
    static_assert(std::is_same<IndexRange<0, 10>, IndexPack<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>>::value, "Error!");
    static_assert(std::is_same<IndexRange<0, 13>, IndexPack<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>>::value, "Error!");
    static_assert(std::is_same<IndexRange<0, 17>, IndexPack<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>>::value, "Error!");
    static_assert(std::is_same<IndexRange<17, 35>, IndexPack<17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34>>::value, "Error!");
}

namespace test_index_at {
    static_assert(IndexRange<0, 100>::At<51>::value == 51, "Error!");
}

namespace test_tuple_index_pack {
    using ITuple = Tuple<int, float, double, Tuple<int, bool>>;
    using STuple = std::tuple<int, float, double, std::tuple<int, bool>>;

    static_assert(std::is_same<TupleIndexPack<ITuple>, IndexRange<0, 4>>::value, "Error!");
    static_assert(std::is_same<TupleIndexPack<STuple>, IndexRange<0, 4>>::value, "Error!");
    static_assert(std::is_same<TupleIndexPack<std::pair<int, float>>, IndexRange<0, 2>>::value, "Error!");
}

namespace test_type_at {
    template<std::size_t I, typename... Args> struct A {};
    template<typename... Args> using AA = A<1, int, Args...>;

    static_assert(std::is_same<TypeAt<3, int, float, double, char>, char>::value, "Error!");
    static_assert(std::is_same<TypeAt<0, int>, int>::value, "Error!");
    static_assert(std::is_same<TypeAt<11, int, int, int, int, int, int, int, float, float, float, double, char>, char>::value, "Error!");
    static_assert(std::is_same<TypeAt<11, int, int, int, int, int, int, int, float, float, float, double, char, bool>, char>::value, "Error!");
    static_assert(std::is_same<TrailingTypes<AA, 3, int, float, double, char, char, bool, long>, A<1, int, char, char, bool, long>>::value, "Error!");
    static_assert(std::is_same<TrailingTypes<std::tuple, 9, char, short, int, long, long long, unsigned char, unsigned short, unsigned, unsigned long, unsigned long long, float, double>,
                               std::tuple<unsigned long long, float, double>>::value, "Error!");
}
