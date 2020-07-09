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
#include "test_util.hpp"
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

    struct CountCriticals {
        template<typename CTX, typename C>
        void operator()(const CTX&, const C&) const noexcept {
            ++*count;
        }

        std::size_t* count;
    };

    template<typename NT, typename Core> impl::CuboidCounts run_handle_nonfinal(
        const Core& core, int initial_splits
    ) {
        std::vector<impl::ProofDriverQueueEntry<NT, 4>> qin, qout;
        for(NT w : Splitter<NT>{NT{0,1}, initial_splits}) {
            for(NT x : Splitter<NT>{NT{0,1}, initial_splits}) {
                Array<NT, 4> result{{w, x, NT{0,1}, NT{0,1}}};
                qin.emplace_back(result);
            }
        }
        return core.handle_cuboids_nonfinal(0, qin, &qout);
    }

    template<typename NT, typename Core> impl::CuboidCounts run_handle_final(
        const Core& core, int initial_splits
    ) {
        std::vector<impl::ProofDriverQueueEntry<NT, 4>> qin;
        for(NT w : Splitter<NT>{NT{0,1}, initial_splits}) {
            for(NT x : Splitter<NT>{NT{0,1}, initial_splits}) {
                Array<NT, 4> result{{w, x, NT{0,1}, NT{0,1}}};
                qin.emplace_back(result);
            }
        }
        return core.handle_cuboids_final(0, qin);
    }

    TEST_CASE_TEMPLATE("[ivarp][run prover] Cuboid count in CPU prover core", NT, IFloat, IDouble) {
        using CTX = DefaultContextWithNumberType<NT>;
        using VarSplit = U64Pack<dynamic_subdivision(4,4), dynamic_subdivision(4,4), 8, 8>;
        const auto input = prover_input<CTX, VarSplit>(system);
        using ProverInputType = BareType<decltype(input)>;
        ProverSettings settings;
        settings.generation_count = 2;
        settings.max_iterations_per_node = 0;
        settings.thread_count = 1;
        settings = impl::replace_default_settings(settings);
        std::size_t cnt = 0;
        CountCriticals counter{&cnt};

        impl::CPUProverCore<ProverInputType, CountCriticals> core;
        impl::DynamicBoundApplication<typename ProverInputType::RuntimeBoundTable, CTX> dba(&input.runtime_bounds);
        core.replace_default_settings(settings);
        core.set_settings(&settings);
        core.set_on_critical(&counter);
        core.set_runtime_bounds(&input.runtime_bounds);
        core.set_dynamic_bound_application(&dba);
        core.set_runtime_constraints(&input.runtime_constraints);
        core.initialize(1);
        core.initialize_per_thread(0, 1);

        auto cgen1 = run_handle_nonfinal<NT>(core, 4);
        auto cgen2 = run_handle_nonfinal<NT>(core, 16);
        auto cgen3 = run_handle_nonfinal<NT>(core, 64);
        REQUIRE(cgen1.num_critical_cuboids == 0);
        REQUIRE(cgen2.num_critical_cuboids == 0);
        REQUIRE(cgen3.num_critical_cuboids == 0);
        CHECK(cgen1.num_leaf_cuboids == 4 * 4 * 8 * 8);
        CHECK(cgen2.num_leaf_cuboids == 16 * 16 * 8 * 8);
        CHECK(cgen3.num_leaf_cuboids == 64 * 64 * 8 * 8);
        CHECK(cgen1.num_cuboids == 4 * 4 * 8 + 4 * 4 * 8 * 8);
        CHECK(cgen2.num_cuboids == 16 * 16 * 8 + 16 * 16 * 8 * 8);
        CHECK(cgen3.num_cuboids == 64 * 64 * 8 + 64 * 64 * 8 * 8);

        auto fgen1 = run_handle_final<NT>(core, 4);
        auto fgen2 = run_handle_final<NT>(core, 16);
        auto fgen3 = run_handle_final<NT>(core, 64);
        CHECK(fgen1.num_critical_cuboids == 4 * 4 * 8 * 8);
        CHECK(fgen1.num_leaf_cuboids == 4 * 4 * 8 * 8);
        CHECK(fgen2.num_critical_cuboids == 16 * 16 * 8 * 8);
        CHECK(fgen2.num_leaf_cuboids == 16 * 16 * 8 * 8);
        CHECK(fgen3.num_critical_cuboids == 64 * 64 * 8 * 8);
        CHECK(fgen3.num_leaf_cuboids == 64 * 64 * 8 * 8);
        CHECK(fgen1.num_cuboids == 4 * 4 * 8 + 4 * 4 * 8 * 8);
        CHECK(fgen2.num_cuboids == 16 * 16 * 8 + 16 * 16 * 8 * 8);
        CHECK(fgen3.num_cuboids == 64 * 64 * 8 + 64 * 64 * 8 * 8);
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

	TEST_CASE_TEMPLATE("[ivarp][run prover] Expected criticals - 2 dynamic and 2 static variables, 1 generation", NT, IFloat, IDouble) {
		using CTX = DefaultContextWithNumberType<NT>;
        using VarSplit = U64Pack<dynamic_subdivision(4,4), dynamic_subdivision(4,4), 8, 8>;
        ProverSettings settings;
        settings.generation_count = 1;
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
        REQUIRE(criticals.size() == 16384);
        CHECK(info.num_repeated_nodes == 0);
        CHECK(info.num_leaf_cuboids == 1024 + 2*16384);
        CHECK(info.num_cuboids ==
            4 + 16 + 8 * 16 + 64 * 16 + // first generation
            4 * 16 + 4 * 4 * 16 + // splitting first generation
            256 * 8 + 256 * 64 + // second generation
            256 * 8 + 256 * 64   // second generation, criticals
        );
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
        CHECK(info.num_cuboids ==
            4 + 16 + 8 * 16 + 64 * 16 + // first generation
            4 * 16 + 4 * 4 * 16 + // splitting first generation
            256 * 8 + 256 * 64 + // second generation
            4 * 256 + 4 * 4 * 256 + // splitting second generation
            8 * 4096 + 64 * 4096 + // third generation
            8 * 4096 + 64 * 4096 // third generation, criticals
        );
    }
}

