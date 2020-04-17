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
#include "../test_util.hpp"
#include "cuda_test_util.cuh"
#include "ivarp/run_prover.hpp"
#include "ivarp/critical_collector.hpp"

namespace {
using namespace ivarp;

template<typename IT> Array<IT,2> expected_critical_ij(std::size_t x_index, std::size_t y_index) {
    using NT = typename IT::NumberType;

    IT xrange = convert_number<IT>(IRational{rational(x_index, 16), rational(x_index+1, 16)});
    NT ymax = xrange.ub();
    NT ymax2 = 3 - xrange.lb();
    if(ymax2 < ymax) {
        ymax = ymax2;
    }
    NT miny = (ymax / 4) * y_index;
    NT maxy = (ymax / 4) * (y_index+1);
    if(miny > 0 && xrange.lb() < miny) {
        xrange.set_lb(miny);
    }
    Array<IT,2> result;
    result[0] = xrange;
    result[1] = IT{miny, maxy};
    return result;
}


TEST_CASE_TEMPLATE("[ivarp][run prover][cuda] Simple expected critical test", IT, IFloat, IDouble) {
    using CTX = DefaultContextWithNumberType<IT>;
    const auto x = args::x0;
    const auto y = args::x1;
    const auto cs = constraint_system(
        variable(x, "x", 0_Z, 2_Z), variable(y, "y",  0_Z, 2_Z),
        x >= y, x <= 3_Z - y, y <= 3_Z - x
    );
    using VarSplit = U64Pack<dynamic_subdivision(8,2), 4>;
    const auto in = prover_input<CTX, VarSplit>(cs);
    using PInput = BareType<decltype(in)>;

    std::mutex lock;
    CriticalCollection<PInput> criticals;
    CriticalCollector<PInput> collector(&criticals, &lock);

    ProverSettings settings;
    settings.generation_count = 2;
    settings.max_iterations_per_node = 1;
    ProofInformation info;
    REQUIRE(info.num_cuboids == 0);
    REQUIRE(info.num_leaf_cuboids == 0);
    REQUIRE(info.num_critical_cuboids == 0);
    REQUIRE(!run_prover(in, collector, &info, settings));
    std::sort(criticals.begin(), criticals.end(), CuboidLexicographicalCompare<IT>{});
    std::size_t first_gen = 8 + 8 * 4;
    std::size_t second_gen = 16 + 16 * 4;
    std::size_t third_gen = 32 + 32 * 4;
    std::size_t third_gen_rep = 32 * 4;
    REQUIRE(info.num_repeated_nodes == 0);
    REQUIRE(info.num_critical_cuboids == 128);
    REQUIRE(info.num_cuboids == first_gen+second_gen+third_gen+third_gen_rep);
    REQUIRE(info.num_leaf_cuboids == 8*4 + 16*4 + 2*32*4);
    REQUIRE(criticals.size() == info.num_critical_cuboids);

    for(std::size_t i = 0, ind = 0; i < 32; ++i) {
        for(std::size_t j = 0; j < 4; ++j, ++ind) {
            auto expected = expected_critical_ij<IT>(i,j);
            REQUIRE_SAME(expected[0], criticals[ind][0]);
            REQUIRE_SAME(expected[1], criticals[ind][1]);
        }
    }
}

}
