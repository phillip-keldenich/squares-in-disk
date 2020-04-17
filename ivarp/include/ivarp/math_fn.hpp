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
// Created by Phillip Keldenich on 2019-09-23.
//

#pragma once

// Standard library prerequisites.
#include <memory>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <type_traits>
#include <string>
#include <iostream>
#include <sstream>
#include <unordered_map>

// IVARP prerequisites.
#include "ivarp/metaprogramming.hpp"
#include "ivarp/array.hpp"
#include "ivarp/tuple.hpp"
#include "ivarp/number.hpp"
#include "ivarp/bool.hpp"
#include "ivarp/context.hpp"
#include "ivarp/bound_direction.hpp"

// Forward declarations.
#include "math_fn/fwd.hpp"

// Implementation: Expression & Predicate template definitions.
#include "math_fn/expression.hpp"
#include "math_fn/math_pred/math_pred_base.hpp"
#include "math_fn/args.hpp"
#include "math_fn/unary_template.hpp"
#include "math_fn/binary_template.hpp"
#include "math_fn/ternary_template.hpp"
#include "math_fn/n_ary_template.hpp"
#include "math_fn/custom_function_template.hpp"
#include "math_fn/constant.hpp"
#include "math_fn/math_pred/bool_constant.hpp"
#include "math_fn/math_pred/unary_template.hpp"
#include "math_fn/math_pred/binary_template.hpp"
#include "math_fn/math_pred/n_ary_template.hpp"
#include "ivarp/refactor_constraint_system/var_bounds.hpp"
#include "math_fn/compile_time_bounds/bounded_expression.hpp"
#include "math_fn/compile_time_bounds/bounded_pred.hpp"

// Implementation: Operation definitions.
#include "math_fn/math_pred/unary_ops.hpp"
#include "math_fn/math_pred/binary_ops.hpp"
#include "math_fn/math_pred/n_ary_ops.hpp"
#include "math_fn/unary_ops.hpp"
#include "math_fn/binary_ops.hpp"
#include "math_fn/ternary_ops.hpp"
#include "math_fn/n_ary_ops.hpp"
#include "math_fn/unary_fns.hpp"
#include "math_fn/fixed_pow.hpp"
#include "math_fn/custom_function.hpp"

// Tag functions.
#include "math_fn/compile_time_bounds/is_bounded.hpp"
#include "math_fn/tag_traits.hpp"
#include "math_fn/tag_of.hpp"

// Meta-functions.
#include "math_fn/math_meta_fn.hpp"
#include "math_fn/math_meta_eval.hpp"
#include "math_fn/has_interval_constants.hpp"
#include "math_fn/preserves_rationality.hpp"
#include "math_fn/numargs.hpp"
#include "math_fn/depends_on_arg.hpp"
#include "math_fn/children.hpp"
#include "math_fn/replace_args.hpp"
#include "math_fn/compile_time_bounds/compute_bounds.hpp"

// Implementation: Symbolic calls.
#include "math_fn/symbolic_call.hpp"

// Implementation: Function evaluation.
#include "math_fn/interval_promotion.hpp"
#include "math_fn/eval.hpp"
#include "math_fn/predicate_eval.hpp"

// Implementation: Printing functions.
#include "math_fn/function_printer.hpp"
