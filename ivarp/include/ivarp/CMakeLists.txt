# The code is open source under the MIT license.
# Copyright 2019-2020, Phillip Keldenich, TU Braunschweig, Algorithms Group
# https://ibr.cs.tu-bs.de/alg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_library(__ivarp_headers INTERFACE)

set(IVARP_HDRS_CUDA "")
if(CMAKE_CUDA_COMPILER)
	set(IVARP_HDRS_CUDA with_cuda/ivarp_cuda/error.hpp with_cuda/ivarp_cuda/memory.hpp)
endif()

set(__IVARP_HEADER_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(ivarp_add_header_group GROUP_NAME GROUP_HEADERS)
	set(L "")
	foreach(fname IN LISTS GROUP_HEADERS)
		list(APPEND L "${__IVARP_HEADER_DIR}/${fname}")
	endforeach()
	target_sources(__ivarp_headers INTERFACE ${L})
	source_group("ivarp\\${GROUP_NAME}" FILES ${L})
endfunction()

set(IVARP_HDRS_SUPPORT cuda.hpp cuda_support/macros.hpp cuda_support/stdcpp.hpp
		array.hpp tuple.hpp symbol_export.hpp atomic_float.hpp cuda_device_discovery.hpp with_cuda/ivarp_cuda/cuda_device_discovery.hpp
)
ivarp_add_header_group(support "${IVARP_HDRS_SUPPORT}")

set(IVARP_HDRS_METAPROGRAMMING metaprogramming.hpp metaprogramming/allof_oneof.hpp metaprogramming/filter_index_pack.hpp
		metaprogramming/index_add_offset.hpp metaprogramming/index_at.hpp metaprogramming/index_concat.hpp
		metaprogramming/index_pack.hpp metaprogramming/index_range.hpp metaprogramming/minof_maxof.hpp
		metaprogramming/tuple_index_pack.hpp metaprogramming/type_at.hpp metaprogramming/predicate_not.hpp
		metaprogramming/bare_type.hpp metaprogramming/merge_index_packs.hpp
)
ivarp_add_header_group(metaprogramming "${IVARP_HDRS_METAPROGRAMMING}")

set(IVARP_HDRS_NUMBER bool.hpp number/bounded_rational.hpp number/decimal_literal.hpp number/device_compat.hpp
		number/exact_less_than.hpp number/fixed_point_bounds.hpp number/fixed_point_bounds_cos.hpp
		number/fixed_point_bounds_impl.hpp number/fixed_point_bounds_sin.hpp number/fixed_point_bounds_sqrt.hpp
		number/float_interval_addsub.hpp number/float_interval_div.hpp number/float_interval_mul.hpp
		number/float_interval_ops.hpp number/float_interval_print.hpp number/fwd.hpp number/int_literal.hpp
		number/interval.hpp number/interval_comparisons.hpp number/interval_div.hpp number/literals.hpp
		number/minmax.hpp number/mpfr.hpp number/rational.hpp number/traits.hpp number/type_conversions.hpp
		rounding.hpp
)
ivarp_add_header_group(number "${IVARP_HDRS_NUMBER}")

set(IVARP_HDRS_MATH_FN math_fn.hpp math_fn/math_meta_fn.hpp math_fn/unary_template.hpp math_fn/depends_on_arg.hpp
		math_fn/n_ary_ops.hpp math_fn/preserves_rationality.hpp math_fn/expression.hpp math_fn/math_meta_eval.hpp
		math_fn/constant.hpp math_fn/tag_traits.hpp math_fn/symbolic_call.hpp math_fn/numargs.hpp
		math_fn/interval_sqrt.hpp math_fn/eval.hpp math_fn/fwd.hpp math_fn/compile_time_bounds/is_bounded.hpp
		math_fn/compile_time_bounds/bounded_expression.hpp math_fn/compile_time_bounds/compute_bounds.hpp
		math_fn/compile_time_bounds/bounded_pred.hpp math_fn/custom_function.hpp math_fn/ternary_ops.hpp
		math_fn/binary_template.hpp math_fn/predicate_eval/fwd.hpp math_fn/predicate_eval/custom.hpp
		math_fn/predicate_eval/unary.hpp math_fn/predicate_eval/n_ary.hpp math_fn/predicate_eval/binary.hpp
		math_fn/predicate_eval/basic.hpp math_fn/predicate_eval.hpp math_fn/unary_fns.hpp math_fn/cuda_constant.hpp
		math_fn/children.hpp math_fn/fixed_pow.hpp math_fn/unary_ops.hpp math_fn/ternary_template.hpp
		math_fn/custom_function_template.hpp math_fn/constant/as.hpp math_fn/constant/fwd.hpp
		math_fn/constant/constant_template.hpp math_fn/constant/cuda_constant.hpp
		math_fn/constant/ensure_expr_or_pred.hpp math_fn/constant/implicitly_convertible.hpp
		math_fn/constant/number_to_constant.hpp math_fn/interval_promotion.hpp math_fn/replace_args.hpp
		math_fn/binary_ops.hpp math_fn/eval/custom.hpp math_fn/eval/if_then_else.hpp math_fn/eval/number_type.hpp
		math_fn/eval/unary.hpp math_fn/eval/n_ary.hpp math_fn/eval/binary.hpp math_fn/eval/ternary.hpp
		math_fn/eval/basic.hpp math_fn/math_pred/unary_template.hpp math_fn/math_pred/n_ary_ops.hpp
		math_fn/math_pred/bool_constant.hpp math_fn/math_pred/binary_template.hpp math_fn/math_pred/unary_ops.hpp
		math_fn/math_pred/math_pred_base.hpp math_fn/math_pred/binary_ops.hpp math_fn/math_pred/n_ary_template.hpp
		math_fn/tag_of.hpp math_fn/args.hpp math_fn/n_ary_template.hpp math_fn/has_interval_constants.hpp
		context.hpp cuda_transformer.hpp math_fn/function_printer.hpp
)
ivarp_add_header_group(math_fn "${IVARP_HDRS_MATH_FN}")

set(IVARP_HDRS_BDA bound_dependency_analysis.hpp bound_dependency_analysis/cbd_args.hpp
		bound_dependency_analysis/cbd_arithmetic_ops.hpp bound_dependency_analysis/cbd_bounded.hpp
		bound_dependency_analysis/cbd_constants.hpp bound_dependency_analysis/cbd_meta_eval_tag.hpp
		bound_dependency_analysis/cbd_simple_deps.hpp bound_dependency_analysis/compute_bound_dependencies.hpp
)
ivarp_add_header_group(bound_dependency_analysis "${IVARP_HDRS_BDA}")

set(IVARP_HDRS_CTB compile_time_bounds.hpp compile_time_bounds/initial_bounds.hpp)
ivarp_add_header_group(compile_time_bounds "${IVARP_HDRS_CTB}")

set(IVARP_HDRS_CONSTANT_FOLDING constant_folding.hpp constant_folding/children.hpp
		constant_folding/constant_folding_expr.hpp constant_folding/constant_folding_expr_eval.hpp
		constant_folding/constant_folding_pred.hpp constant_folding/constant_folding_pred_eval.hpp
		constant_folding/fold_constants.hpp
)
ivarp_add_header_group(constant_folding "${IVARP_HDRS_CONSTANT_FOLDING}")

set(IVARP_HDRS_CONSTRAINT_REFORMULATION constraint_reformulation.hpp
		constraint_reformulation/reformulate_binary_pred.hpp constraint_reformulation/reformulate_negation.hpp)
ivarp_add_header_group(constraint_reformulation "${IVARP_HDRS_CONSTRAINT_REFORMULATION}")

set(IVARP_HDRS_PROVER prover.hpp critical_printer.hpp critical_reducer.hpp default_progress_observer.hpp
		prover/check_predicate_to_constraints.hpp prover/constraint_propagation.hpp
		prover/constraints.hpp prover/cuboid_queue.hpp prover/dynamic_entry.hpp prover/proof_runner.hpp
		prover/proof_runner_impl.hpp prover/prover.hpp prover/prover_cpu.hpp prover/prover_thread.hpp
		prover/variable.hpp run_prover.hpp
)
ivarp_add_header_group(prover "${IVARP_HDRS_PROVER}")

set(IVARP_HDRS_CONSTRAINT_SYSTEM constraint_system.hpp variable_description.hpp
		constraint_system/extract_bound_constraints.hpp constraint_system/extract_ct_bounds.hpp)
ivarp_add_header_group(constraint_system "${IVARP_HDRS_CONSTRAINT_SYSTEM}")

set(IVARP_HDRS_REFACTOR_CS refactor_constraint_system.hpp refactor_constraint_system/var_bounds.hpp
		refactor_constraint_system/runtime_bound_table.hpp
		refactor_constraint_system/constraints_to_bounds.hpp
		refactor_constraint_system/bound_and_simplify/tag_based.hpp
		refactor_constraint_system/bound_and_simplify/args.hpp
		refactor_constraint_system/bound_and_simplify/bounded.hpp
		refactor_constraint_system/bound_and_simplify/custom.hpp
		refactor_constraint_system/bound_and_simplify/constant.hpp
		refactor_constraint_system/filter_constraints.hpp refactor_constraint_system/bound_and_simplify.hpp
		refactor_constraint_system/runtime_constraint_table.hpp refactor_constraint_system/labeled_bound.hpp
		refactor_constraint_system/maybe_bound.hpp refactor_constraint_system/refactor_constraints_impl.hpp
		refactor_constraint_system/constraints_to_bounds/rewriters/negation.hpp
		refactor_constraint_system/constraints_to_bounds/rewriters/relational_ops.hpp
		refactor_constraint_system/constraints_to_bounds/rewrite_to_bound.hpp
		refactor_constraint_system/constraints_to_bounds/rewrite_failed.hpp
		refactor_constraint_system/constraints_to_bounds/compile_time_bound.hpp
		prover_input.hpp prover_input/make_prover_input_impl.hpp)
ivarp_add_header_group(refactor_constraint_system "${IVARP_HDRS_REFACTOR_CS}")
