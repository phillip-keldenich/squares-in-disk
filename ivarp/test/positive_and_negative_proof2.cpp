#ifdef __CUDACC__
#define IVARP_POSITIVE_NEGATIVE_PROOF2_NAME "[ivarp][prover][cuda] Full Prover Test 2 - "
#else
#define IVARP_POSITIVE_NEGATIVE_PROOF2_NAME "[ivarp][prover] Full Prover Test 2 - "
#endif

#include "test_util.hpp"
#include "ivarp/run_prover.hpp"
#include "ivarp/critical_printer.hpp"

namespace {
	using namespace ivarp;

	const auto w = args::x0;
	const auto x = args::x1;
	const auto y = args::x2;
	const auto z = args::x3;

	const auto f = -w * (fixed_pow<3>(x) - 2_Z*square(y) + sqrt(y)) + w/z;
	const auto system_unsat = constraint_system(
			variable(w, "w", -2_Z, 2_Z), variable(x, "x", -2_Z, 2_Z),
			variable(y, "y", 0_Z, 5_Z), variable(z, "z", 0.5_X, 8_Z),
			f > 111.927865_X, x <= w, z >= y
	);
	const auto system_sat = constraint_system(
			variable(w, "w", -2_Z, 2_Z), variable(x, "x", -2_Z, 2_Z),
			variable(y, "y", 0_Z, 5_Z), variable(z, "z", 0.5_X, 8_Z),
			f > 111.927864_X, x <= w, z >= y
	);

	using VarSplit = U64Pack<dynamic_subdivision(8,4), dynamic_subdivision(8,4), 128, 128>;

	TEST_CASE_TEMPLATE(IVARP_POSITIVE_NEGATIVE_PROOF2_NAME "Positive", NT, IDouble) {
		using CTX = DefaultContextWithNumberType<NT>;
		const auto input = prover_input<CTX, VarSplit>(system_unsat);
		ProverSettings settings;
		ProofInformation info;
		settings.generation_count = 16;
		settings.max_iterations_per_node = 0;
		const auto printer = critical_printer(std::cerr, system_sat, printable_expression("f(w,x,y,z)", f));
		REQUIRE(run_prover(input, printer, &info, settings));
		REQUIRE(info.num_cuboids > 0);
		REQUIRE(info.num_critical_cuboids == 0);
	}

	struct Handler {
		template<typename Context, typename V> void operator()(const Context&, const V&) const noexcept {
			++(*count);
		}

		std::atomic<std::size_t> *count;
	};

	TEST_CASE_TEMPLATE(IVARP_POSITIVE_NEGATIVE_PROOF2_NAME "Negative", NT, IDouble) {
		using CTX = DefaultContextWithNumberType<NT>;
		const auto input = prover_input<CTX, VarSplit>(system_sat);
		ProverSettings settings;
		ProofInformation info;
		settings.generation_count = 16;
		settings.max_iterations_per_node = 0;
		std::atomic<std::size_t> count(0);
		Handler h{&count};
		
		REQUIRE(!run_prover(input, h, &info, settings));
		REQUIRE(info.num_cuboids > 0);
		REQUIRE(info.num_critical_cuboids > 0);
		REQUIRE(info.num_critical_cuboids == count.load());
	}
}

