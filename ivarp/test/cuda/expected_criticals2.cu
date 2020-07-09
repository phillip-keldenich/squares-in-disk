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
// Created by Phillip Keldenich on 30.04.20.
//
#include "../test_util.hpp"
#include "ivarp/run_prover.hpp"
#include "ivarp/critical_collector.hpp"

namespace {
	using namespace ivarp;
    using namespace ivarp::args;
    const auto w = x0;
    const auto x = x1;
    const auto y = x2;
    const auto z = x3;
    const auto f = w + x + y + z;
    const auto system = constraint_system(
        variable(w, "w", 0_Z, 1_Z), variable(x, "x", 0_Z, 1_Z),
        variable(y, "y", 0_Z, 1_Z), variable(z, "z", 0_Z, 1_Z),
        w + x + y + z >= 0
    );

    template<typename NT, typename CT>
        static bool check_criticals(const CT& criticals)
    {
        std::size_t count = 0;
        for(NT w : Splitter<NT>{NT{0,1}, 64}) {
            for(NT x : Splitter<NT>{NT{0,1}, 64}) {
                for(NT y : Splitter<NT>{NT{0,1}, 8}) {
                    for(NT z : Splitter<NT>{NT{0,1}, 8}) {
                        REQUIRE_SAME(criticals[count][0], w);
                        REQUIRE_SAME(criticals[count][1], x);
                        REQUIRE_SAME(criticals[count][2], y);
                        REQUIRE_SAME(criticals[count][3], z);
                        ++count;
                    }
                }
            }
        }
        return true;
    }


	TEST_CASE_TEMPLATE("[ivarp][run prover] Expected criticals - 2 dynamic and 2 static variables, no generations", NT, IFloat, IDouble) {
		using CTX = DefaultContextWithNumberType<NT>;
        using VarSplit = U64Pack<dynamic_subdivision(4,4), dynamic_subdivision(4,4), 8, 8>;
        ProverSettings settings;
        settings.generation_count = 0;
		settings.max_iterations_per_node = 0;
		ProofInformation info;
		CHECK(info.num_cuboids == 0);
        const auto input = prover_input<CTX, VarSplit>(system);
        using ProverInputType = BareType<decltype(input)>;
        std::mutex lock;
        CriticalCollection<ProverInputType> criticals;
        CriticalCollector<ProverInputType> collector(&criticals, &lock);
        REQUIRE(!run_prover(input, collector, &info, settings));
        REQUIRE(criticals.size() == info.num_critical_cuboids);
        REQUIRE(criticals.size() == 1024);
        CHECK(info.num_repeated_nodes == 0);
        CHECK(info.num_leaf_cuboids == 2*1024);
        CHECK(info.num_cuboids == 1172 + 1152);
	}

    TEST_CASE_TEMPLATE("[ivarp][run prover] Expected criticals - 2 dynamic and 2 static variables", NT, IFloat, IDouble) {
        using CTX = DefaultContextWithNumberType<NT>;
        using VarSplit = U64Pack<dynamic_subdivision(4,4), dynamic_subdivision(4,4), 8, 8>;
        ProverSettings settings;
        settings.generation_count = 2;
        settings.max_iterations_per_node = 0;
        ProofInformation info;
		CHECK(info.num_cuboids == 0);
        const auto input = prover_input<CTX, VarSplit>(system);
        using ProverInputType = BareType<decltype(input)>;
        std::mutex lock;
        CriticalCollection<ProverInputType> criticals;
        CriticalCollector<ProverInputType> collector(&criticals, &lock);
        REQUIRE(!run_prover(input, collector, &info, settings));
        std::sort(criticals.begin(), criticals.end(), CuboidLexicographicalCompare<NT>{});
        REQUIRE(criticals.size() == info.num_critical_cuboids);
        REQUIRE(criticals.size() == 262144);
        REQUIRE(check_criticals<NT>(criticals));
        CHECK(info.num_repeated_nodes == 0);
        CHECK(info.num_leaf_cuboids == 1024 + 16384 + 262144*2);
        CHECK(info.num_cuboids == 1172 + 18704 + 299072 + 294912);
    }
}

