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
// Created by Phillip Keldenich on 13.11.19.
//

#pragma once

namespace ivarp {
    template<typename Variables, typename Constraints> class ConstraintPropagation {
    public:
        static_assert(IsVariables<Variables>::value, "Invalid Variables given to ConstraintPropagation!");
        static_assert(IsConstraints<Constraints>::value, "Invalid Constraints given to ConstraintPropagation!");

        explicit ConstraintPropagation(const Variables* variables, const Constraints* constraints) noexcept :
            variables(variables),
            constraints(constraints)
        {}

        struct PropagationResult {
            bool empty;
        };

        /// Compute the bounds of the variable with index NumVarsPresent in the given array.
        /// The result informs about whether the bounds computed for the new variable are empty.
        template<std::size_t NumVarsPresent, typename Context, typename ArrayType>
            PropagationResult compute_new_var_bounds(ArrayType& args) const
        {
            // Begin with bounds specified with the variable.
            const auto& variable = variables->vars[IVARP_IND(NumVarsPresent)];
            auto vbounds = variable.compute_bounds(Context{}, args);
            assert(!vbounds.possibly_undefined());
            if(vbounds.empty()) {
                return PropagationResult{true};
            }
			ivarp::get<NumVarsPresent>(args) = vbounds;
            return restrict_new_var_bounds_constraints<NumVarsPresent, Context>(args);
        }

        /// Restrict the bound of the variable with index NumVarsPresent in the given array.
        /// Use this instead of compute_new_var_bounds if the given array already contains valid bounds for
        /// the variable for which to compute new bounds.
        /// The result informs about whether the bounds for the new variable are empty.
        template<std::size_t NumVarsPresent, typename Context, typename ArrayType>
            PropagationResult restrict_new_var_bounds(ArrayType& args) const
        {
            // Begin with bounds specified with the variable.
            const auto& variable = ivarp::get<NumVarsPresent>(variables->vars);
            auto vbounds = variable.compute_bounds(Context{}, args);
            assert(!vbounds.possibly_undefined());
            if(vbounds.empty()) {
                return {true};
            }
            if(ivarp::get<NumVarsPresent>(args).restrict_lower_bound(vbounds).result_empty ||
               ivarp::get<NumVarsPresent>(args).restrict_upper_bound(vbounds).result_empty)
            {
                return {true};
            }
            return restrict_new_var_bounds_constraints<NumVarsPresent, Context>(args);
        }

        /// Restrict the bounds of the given variable, given NumVarsPresent variable bounds in args.
        template<std::size_t RestrictVar, std::size_t NumVarsPresent, typename Context, typename ArgArray>
            PropagationResult restrict_var_bounds(ArgArray& args) const
        {
            // Again, begin by applying the variable bounds.
            const auto& variable = ivarp::get<RestrictVar>(variables->vars);
            auto vbounds = variable.compute_bounds(Context{}, args);
            assert(!vbounds.possibly_undefined());
            if(vbounds.empty()) {
                return PropagationResult{true};
            }
            if(ivarp::get<RestrictVar>(args).restrict_lower_bound(vbounds).result_empty ||
               ivarp::get<RestrictVar>(args).restrict_upper_bound(vbounds).result_empty)
            {
                return PropagationResult{true};
            }
            return restrict_var_bounds_all_constraints<RestrictVar, NumVarsPresent, Context>(args);
        }


        /// Run propagation after we have done a split on the variable SplitVar.
        /// This begins by applying all newly applicable bounds; if any variable bounds are
        /// changed in the process, bounds that depend on the changed variable are reconsidered.
        /// For each variable, propagation is not performed more than ApplicationsPerVar times.
        /// Finally, if there were any bound changes, all applicable constraints are tried for pruning;
        /// otherwise, only newly applicable constraints are tried for pruning.
        template<std::size_t SplitVar, std::size_t ApplicationsPerVar, typename Context,
                 std::size_t NumVarsPresent = SplitVar+1, typename ArgArray>
            PropagationResult dynamic_post_split_var(ArgArray& args) const
        {
            // zero-initialize all application counters; set the counter for SplitVar to 1.
            Array<std::size_t, NumVarsPresent> app_counts = {};
            app_counts[SplitVar] = 1;

            ApplyBoundResult prop_result = propagate_var_bound_changed<SplitVar,
                                                                       ApplicationsPerVar, Context>(args, app_counts);

            if(prop_result.result_empty) {
                return PropagationResult{true};
            }

            if(prop_result.bound_changed) {
                using ApplicableConstraintIndices = typename Constraints::template ApplicableWithVars<NumVarsPresent>;
                return PropagationResult{can_constraints_prune<Context>(args, ApplicableConstraintIndices{})};
            } else {
                using NewConstraintIndices = typename Constraints::template NewlyApplicableWithVars<SplitVar+1>;
                return PropagationResult{can_constraints_prune<Context>(args, NewConstraintIndices{})};
            }
        }

        template<std::size_t SplitVar, std::size_t ApplicationsPerVar, typename Context, typename ArgArray>
            PropagationResult static_post_split_var(ArgArray& args) const
        {
            // zero-initialize all application counters; set the counter for SplitVar to 1.
            Array<std::size_t, SplitVar+1> app_counts = {};
            app_counts[SplitVar] = 1;

            ApplyBoundResult prop_result = propagate_var_bound_changed<SplitVar,
                                                                       ApplicationsPerVar, Context>(args, app_counts);

            if(prop_result.result_empty) {
                return PropagationResult{true};
            }

            using NewConstraintIndices = typename Constraints::template NewlyApplicableWithVars<SplitVar+1>;
            return PropagationResult{can_constraints_prune<Context>(args, NewConstraintIndices{})};
        }

        template<std::size_t NumVarsPresent, typename Context, typename ArgArray>
            bool can_constraints_prune(const ArgArray& args) const
        {
            using Applicable = typename Constraints::template ApplicableWithVars<NumVarsPresent>;
            return can_constraints_prune<Context>(args, Applicable{});
        }

    private:
        const Variables* variables;
        const Constraints* constraints;

        /// Incorporate constraints for bounds and pruning
        /// into the result from restrict_new_var_bounds/compute_new_var_bounds.
        template<std::size_t NumVarsPresent, typename Context, typename ArrayType>
            PropagationResult restrict_new_var_bounds_constraints(ArrayType& args) const
        {
            // Also incorporate constraints.
            using RelevantConstraintIndices = typename Constraints::template NewVariableBoundConstraints<NumVarsPresent>;
            if(restrict_new_var_bounds<NumVarsPresent, Context>(args, RelevantConstraintIndices{}).empty) {
                return PropagationResult{true};
            }

            // Constraints that may not be used as bounds may still be useful to prune.
            using PruneConstraintIndices = typename Constraints::template NewVariablePruneConstraints<NumVarsPresent>;
            return PropagationResult{can_constraints_prune<Context>(args, PruneConstraintIndices{})};
        }

        template<std::size_t RestrictVar, std::size_t NumVarsPresent, typename Context, typename ArgArray>
            PropagationResult restrict_var_bounds_all_constraints(ArgArray& args) const
        {
            using RelevantConstraintIndices = typename Constraints::
                template RestrictBoundConstraints<RestrictVar,NumVarsPresent>;
            return restrict_var_bounds_all_constraints<RestrictVar,NumVarsPresent,Context>(args,
                                                                                           RelevantConstraintIndices{});
        }

        template<std::size_t RestrictVar, std::size_t NumVarsPresent, typename Context, typename ArgArray,
                 std::size_t C1, std::size_t... Cons>
            PropagationResult restrict_var_bounds_all_constraints(ArgArray& args, IndexPack<C1,Cons...>) const
        {
            const auto& ctb = ivarp::get<RestrictVar>(ivarp::get<C1>(constraints->constraints_to_bounds));
            if (ctb.apply_bound(Context{}, *constraints, args).result_empty) {
                return PropagationResult{true};
            } else {
                return restrict_var_bounds_all_constraints<RestrictVar,NumVarsPresent,Context>(args,
                                                                                               IndexPack<Cons...>{});
            }
        }

        template<std::size_t RestrictVar, std::size_t NumVarsPresent, typename Context, typename ArgArray>
            PropagationResult restrict_var_bounds_all_constraints(ArgArray& /*args*/, IndexPack<>) const
        {
            return PropagationResult{false};
        }

        /// Restrict the range of the variable indicated by NewVar using the constraints indicated by the IndexPack.
        /// Each of these constraints must be potentially rewritable to a NewVar bound.
        template<std::size_t NewVar, typename Context, typename ArrayType>
            PropagationResult restrict_new_var_bounds(ArrayType& /*args*/, IndexPack<>) const noexcept
        {
            return PropagationResult{false};
        }
        template<std::size_t NewVar, typename Context, typename ArrayType, std::size_t I1, std::size_t... Inds>
            PropagationResult restrict_new_var_bounds(ArrayType& args, IndexPack<I1,Inds...>) const
        {
            const auto& outer_ctb = ivarp::get<I1>(constraints->constraints_to_bounds);
            const auto& ctb = ivarp::get<NewVar>(outer_ctb);
            if(ctb.apply_bound(Context{}, *constraints, args).result_empty) {
                return PropagationResult{true};
            } else {
                return restrict_new_var_bounds<NewVar, Context>(args, IndexPack<Inds...>{});
            }
        }

        /// Check whether the constraints indicated by the indices in the IndexPack can prune off the given cuboid.
        template<typename Context, typename ArrayType>
            bool can_constraints_prune(const ArrayType& /*args*/, IndexPack<>) const noexcept
        {
            return false;
        }
        template<typename Context, typename ArrayType, std::size_t I1, std::size_t... Inds>
            bool can_constraints_prune(const ArrayType& args, IndexPack<I1,Inds...>) const
        {
            const auto& constraint = ivarp::get<I1>(constraints->constraints);
            if(!possibly(constraint.template array_evaluate<Context>(args))) {
                return true;
            }
            return can_constraints_prune<Context>(args, IndexPack<Inds...>{});
        }

        /// Propagate changes to the bounds of the given variable.
        template<std::size_t ChangedVar, std::size_t ApplicationsPerVar, typename Context,
                 std::size_t NumVars, typename ArgArray>
            ApplyBoundResult propagate_var_bound_changed(ArgArray& args,
                                                         Array<std::size_t, NumVars>& app_counts) const
        {
            Array<bool, NumVars> changed = {};

            using ConstraintIndices = typename Constraints::template BoundPropagationConstraints<NumVars, ChangedVar>;
            ApplyBoundResult result = propagate_var_bound_changed<ChangedVar, ApplicationsPerVar,
                                                                  Context>(args, app_counts, changed,
                                                                           ConstraintIndices{});
            if(result.bound_changed) {
                if(result.result_empty) {
                    return ApplyBoundResult{true,true};
                }
                result |= propagate_handle_changes<ApplicationsPerVar, Context>(args, app_counts, changed,
                                                                                IndexRange<0,NumVars>{});
            }
            return result;
        }

        /// Use the specified constraint for propagation.
        template<typename ConstraintType, std::size_t ConstraintIndex, typename Context, std::size_t NumVars,
                 typename ArgArray, std::size_t V1, std::size_t... Vars>
            ApplyBoundResult propagate_use_constraint(const ConstraintType& constraint, ArgArray& args,
                                                      Array<bool,NumVars>& changed,
                                                      IndexPack<V1, Vars...>) const
        {
            const auto& ctb = ivarp::get<V1>(ivarp::get<ConstraintIndex>(constraints->constraints_to_bounds));
            ApplyBoundResult result = ctb.apply_bound(Context{}, *constraints, args);
            if(result.result_empty) {
                return ApplyBoundResult{true,true};
            }
            if(result.bound_changed) {
				ivarp::get<V1>(changed) = true;
            }
            return result |
                propagate_use_constraint<ConstraintType, ConstraintIndex, Context>(constraint, args,
                                                                                   changed, IndexPack<Vars...>{});
        }

        /// Handle the case where we have considered all variables we have to consider.
        template<typename ConstraintType, std::size_t ConstraintIndex, typename Context, std::size_t NumVars,
                 typename ArgArray>
            ApplyBoundResult propagate_use_constraint(const ConstraintType& /*constraint*/,
                                                      ArgArray& /*args*/,
                                                      Array<bool,NumVars>& /*changed*/, IndexPack<>) const
        {
            return ApplyBoundResult{false,false};
        }

        /// After handling propagation on a variable, invoke propagation on the changed variables.
        template<std::size_t ApplicationsPerVar, typename Context, std::size_t NumVars, typename ArgArray,
                 std::size_t I1, std::size_t... Inds>
            ApplyBoundResult propagate_handle_changes(ArgArray& args, Array<std::size_t, NumVars>& app_counts,
                                                      const Array<bool, NumVars>& changed, IndexPack<I1,Inds...>) const
        {
            // if the first remaining variable changed and we have not yet applied propagation to it
            // at least ApplicationsPerVar times, run propagation on it.
            if(ivarp::get<I1>(changed) && ivarp::get<I1>(app_counts)++ < ApplicationsPerVar) {
                ApplyBoundResult result =
                    propagate_var_bound_changed<I1, ApplicationsPerVar, Context>(args, app_counts);

                if(result.result_empty) {
                    return ApplyBoundResult{true,true};
                }
                return result | propagate_handle_changes<ApplicationsPerVar, Context>(args, app_counts, changed,
                                                                                      IndexPack<Inds...>{});
            }

            return propagate_handle_changes<ApplicationsPerVar, Context>(args, app_counts, changed,
                                                                         IndexPack<Inds...>{});
        }

        /// This handles the case where we are done with all potential changes.
        template<std::size_t ApplicationsPerVar, typename Context, std::size_t NumVars, typename ArgArray>
            ApplyBoundResult propagate_handle_changes(ArgArray& /*args*/,
                                                      Array<std::size_t, NumVars>& /*app_counts*/,
                                                      const Array<bool, NumVars>& /*changed*/, IndexPack<>) const
        {
            return ApplyBoundResult{false,false};
        }

        template<std::size_t ChangedVar, std::size_t ApplicationsPerVar, typename Context,
                 std::size_t NumVars, typename ArgArray, std::size_t C1, std::size_t... ConstraintIds>
            ApplyBoundResult propagate_var_bound_changed(ArgArray& args, Array<std::size_t, NumVars>& app_counts,
                                                         Array<bool, NumVars>& changed,
                                                         IndexPack<C1, ConstraintIds...>) const
        {
            using ConstraintType = typename Constraints::template ByIndex<C1>;
            using VarIndices = typename Constraints::
                                   template RewritableVariables<ConstraintType, NumVars, ChangedVar>;

            const ConstraintType& constraint = ivarp::get<C1>(constraints->constraints);
            ApplyBoundResult result = propagate_use_constraint<ConstraintType, C1, Context>(constraint, args,
                                                                                            changed, VarIndices{});
            if(result.result_empty) {
                return ApplyBoundResult{true,true};
            }
            return result | propagate_var_bound_changed<ChangedVar, ApplicationsPerVar,
                                                        Context>(args, app_counts, changed,
                                                                 IndexPack<ConstraintIds...>{});
        }

        template<std::size_t ChangedVar, std::size_t ApplicationsPerVar, typename Context,
                 std::size_t NumVars, typename ArgArray>
            ApplyBoundResult propagate_var_bound_changed(ArgArray& /*args*/, Array<std::size_t, NumVars>& /*app_counts*/,
                                                         Array<bool, NumVars>& /*changed*/,
                                                         IndexPack<>) const
        {
            return ApplyBoundResult{false,false};
        }
    };

    template<typename Vars, typename Constrs> static inline ConstraintPropagation<Vars, Constrs>
        constraint_propagation(const Vars* vars, const Constrs* constrs) noexcept
    {
        return ConstraintPropagation<Vars, Constrs>{vars,constrs};
    }
}
