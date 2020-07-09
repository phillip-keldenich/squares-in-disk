add_executable(ivarp_unit_tests interval.cpp unit_tests.cpp metaprogramming.cpp test_convert_number.cpp literals.cpp
		fixed_point_bounds.cpp numargs_test.cpp interval_sincos.cpp interval_addsub.cpp rng.cpp
		interval_mul.cpp test_util.hpp interval_div.cpp powint.cpp interval_sqrt.cpp math_pred.cpp exact_less_than.cpp
		splitter.cpp constraint_propagation.cpp simple_positive_proof.cpp positive_and_negative_proof1.cpp positive_and_negative_proof2.cpp
		symbolic_call.cpp  if_then_else.cpp custom_function.cpp constant_folding_custom_function.cpp tuple.cpp
		fn_eval.cpp tag_of.cpp compile_time_bounds.cpp initial_compile_time_bounds.cpp bound_and_simplify.cpp
		rewrite_to_bounds.cpp bound_dependency_analysis.cpp constraint_system.cpp prover_input.cpp run_prover.cpp
		function_printer.cpp issues/issue8.cpp factorize.cpp expected_criticals.cpp expected_criticals2.cpp math.cpp)

target_link_libraries(ivarp_unit_tests PRIVATE ivarp doctest util::enable_warnings util::debug_use_asan)

if(CMAKE_CUDA_COMPILER)
	target_sources(ivarp_unit_tests PRIVATE cuda/intervals.cu cuda/device_tuple.cu
			                                cuda/exact_less_than.cu cuda/critical_reducer.cu
			                                cuda/fn_eval.cu cuda/expected_criticals.cu cuda/expected_criticals2.cu
				                            cuda/positive_and_negative_proof1.cu
											cuda/positive_and_negative_proof2.cu
											cuda/math.cu cuda/grid_dimensions.cu
											issues/issue9.cu)
endif()

target_enable_lto(ivarp_unit_tests)
