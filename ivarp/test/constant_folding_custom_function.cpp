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
#include <doctest/doctest_fixed.hpp>
#include "ivarp/math_fn.hpp"
#include "ivarp/constant_folding.hpp"
#include "test_util.hpp"

using namespace ivarp;

namespace {
	const auto x = args::x0;

	const auto custom_add = [] (const auto& /*ctx*/, const auto& a1, const auto& a2) -> auto {
        return a1 + a2;
    };

	template<typename N> Interval<N> broaden(const N& n) {
		return Interval<N>(n-0.5f,n+0.5f);
	}

	template<typename N> Interval<N> broaden(const Interval<N>& n) {
		return Interval<N>(n.lb()-0.5f,n.ub()+0.5f);
	}

	const auto custom_broaden = [] (const auto& /*ctx*/, const auto& a1) -> auto {
		return broaden(a1);
	};

	const auto custom_add_foldable = custom_function_context_as_arg(custom_add, 100_Z, 150_Z/10_Z);
	const auto custom_add_unfoldable = custom_function_context_as_arg(custom_add, x, 100_Z);
	const auto custom_broaden_foldable = custom_function_context_as_arg(custom_broaden, 5_Z);
	const auto custom_broaden_unfoldable = custom_function_context_as_arg(custom_broaden, x);

	using FoldableType = std::decay_t<decltype(custom_add_foldable)>;
	using UnfoldableType = std::decay_t<decltype(custom_add_unfoldable)>;
	using FoldableType2 = std::decay_t<decltype(custom_broaden_foldable)>;
	using UnfoldableType2 = std::decay_t<decltype(custom_broaden_unfoldable)>;

	using FunctorUnf2 = typename UnfoldableType2::FunctorType;
	using CalledWithRationalAndBool2 = typename impl::CallWithRationalAndBool<FunctorUnf2, std::decay_t<decltype(x)>>::Type;

	static_assert(std::is_same<CalledWithRationalAndBool2, IRational>::value, "Should be interval");

	static_assert(CanConstantFold<FoldableType>::value, "Should be foldable!");
	static_assert(!CanConstantFold<UnfoldableType>::value, "Should NOT be foldable!");
	static_assert(CanConstantFold<FoldableType2>::value, "Should be foldable!");
	static_assert(!CanConstantFold<UnfoldableType2>::value, "Should NOT be foldable!");
	static_assert(impl::PreservesRationality<UnfoldableType>::value, "Should preserve rationality!");
	static_assert(!impl::PreservesRationality<UnfoldableType2>::value, "Should NOT preserve rationality!");

	const auto folded = fold_constants(custom_add_foldable);
	static_assert(std::is_same<
	        ConstantFoldedExpr<FoldableType, fixed_point_bounds::min_bound(), fixed_point_bounds::max_bound()>,
	        std::decay_t<decltype(folded)>>::value, "Should be folded expression!");

	const auto bfolded = fold_constants(custom_broaden_foldable);
	static_assert(std::is_same<
	        ConstantFoldedExpr<FoldableType2, fixed_point_bounds::min_bound(), fixed_point_bounds::max_bound()>,
	        std::decay_t<decltype(bfolded)>>::value, "Should be folded expression!");
}

TEST_CASE("[ivarp][constant folding] Constant folding of simple custom function") {
	REQUIRE_SAME(folded.ifloat, IFloat(115));
	REQUIRE_SAME(folded.idouble, IDouble(115));
	Rational r = folded.irational;
	REQUIRE(r == 115);
}

TEST_CASE("[ivarp][constant folding] Constant folding of non-rationality-preserving function") {
	REQUIRE_SAME(bfolded.ifloat, IFloat(4.5f,5.5f));
	REQUIRE_SAME(bfolded.idouble, IDouble(4.5,5.5));
	REQUIRE_SAME(bfolded.irational, IRational(Rational(9,2),Rational(11,2)));
}

