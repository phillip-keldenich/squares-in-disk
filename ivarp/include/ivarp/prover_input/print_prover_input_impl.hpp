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

#pragma once

namespace ivarp {
namespace impl {
    template<typename CTInitialBounds, typename InitialBounds,std::size_t... ArgInds, typename... VarSplitInfo>
    static inline void print_prover_input_vars(
        std::ostream& output, const InitialBounds& ib, IndexPack<ArgInds...>,
        SplitInfoSequence<VarSplitInfo...>,
        FunctionPrinter& fprinter)
    {
        std::size_t args[] = {
            VarSplitInfo::arg...
        };
        std::size_t splits[] = {
            VarSplitInfo::subdivisions...
        };
        std::size_t initial[] = {
            VarSplitInfo::initial...
        };
        std::int64_t ct_lb[] = {
            CTInitialBounds::template At<ArgInds>::lb...
        };
        std::int64_t ct_ub[] = {
            CTInitialBounds::template At<ArgInds>::ub...
        };
        bool dyn[] = {
            VarSplitInfo::is_dynamic...
        };

        std::size_t splitind = 0;
        output << "[Arguments]\n";
        for (std::size_t i = 0; i < sizeof...(ArgInds); ++i) {
            output << "    " << i << ": ";
            if (args[splitind] == i) {
                if (dyn[splitind]) {
                    output << "Dynamic variable '";
                    fprinter.print_arg(output, i);
                    output << "', initial subs: " << initial[splitind]
                           << ", subs per subdivision: " << splits[splitind];
                } else {
                    output << "Static variable '";
                    fprinter.print_arg(output, i);
                    output << "', subs per subdivision: " << splits[splitind];
                }
                ++splitind;
            } else {
                output << "Value '";
                fprinter.print_arg(output, i);
                output << '\'';
            }
            output << ", compile time bounds: [" << fixed_point_bounds::PrintFixedPoint(ct_lb[i])
                   << ", " << fixed_point_bounds::PrintFixedPoint(ct_ub[i]) << "]"
                   << ", initial range: " << ib[i] << std::endl;
        }
    }

    template<typename BoundFunction, BoundDirection Direction>
    static inline void print_bound(std::ostream& output,
                                   std::size_t arg,
                                   const CompileTimeBound<BoundFunction, Direction>& b,
                                   FunctionPrinter& fprinter)
    {
        output << "CT bound: '";
        fprinter.print_arg(output, arg);
        output << "' ";
        switch (b.direction) {
        default:
        case BoundDirection::BOTH:
            output << "== ";
            break;
        case BoundDirection::GEQ:
            output << ">= ";
            break;
        case BoundDirection::LEQ:
            output << "<= ";
            break;
        }
        fprinter.print(output, b.bound);
    }

    template<typename BoundFunction, typename DirectionCheck>
    static inline void print_bound(std::ostream& output,
                                   std::size_t arg,
                                   const MaybeBound<BoundFunction, DirectionCheck>& b,
                                   FunctionPrinter& fprinter)
    {
        output << "Possible bound: '";
        fprinter.print_arg(output, arg);
        output << "' ~= ";
        fprinter.print(output, b.bound);
    }

    template<std::size_t Ind, typename PIType>
    static inline int print_prover_input_bound(std::ostream& output, const PIType& prover_in,
                                               FunctionPrinter& fprinter)
    {
        output << "    " << Ind << ": ";
        const auto& bound = prover_in.runtime_bounds.template get<Ind>();
        print_bound(output, bound.get_target(), bound.get_bound(), fprinter);
        output << std::endl;
        return 0;
    }

    template<typename PIType, std::size_t... Inds>
    static inline void print_prover_input_bounds(std::ostream& output, const PIType& prover_in, 
                                                 FunctionPrinter& fprinter, IndexPack<Inds...>)
    {
        output << "[Bounds]\n";
        ConstructWithAny{print_prover_input_bound<Inds>(output, prover_in, fprinter)...};
    }

    template<typename RCTCell>
    static inline int print_constraint_table_cell(std::ostream& output, FunctionPrinter& fprinter,
                                                  const RCTCell& cell)
    {
        output << "        ";
        fprinter.print(output, cell);
        output << std::endl;
        return 0;
    }

    template<typename RCTRow, std::size_t... CInds>
    static inline int print_constraint_table_row(std::ostream& output, FunctionPrinter& fprinter,
                                                 std::size_t arg_index, const RCTRow& row, IndexPack<CInds...>)
    {
        if(sizeof...(CInds) > 0) {
            output << "     " << (arg_index+1) << " arguments:\n";
        }

        ConstructWithAny{
            print_constraint_table_cell(output, fprinter, ivarp::template get<CInds>(row))...
        };
        return 0;
    }

    template<typename RCT, std::size_t... ArgInds>
    static inline void print_prover_input_constraints(std::ostream& output, FunctionPrinter& fprinter,
                                                      const RCT& rct, IndexPack<ArgInds...>)
    {
        output << "[Constraints]\n";
        ConstructWithAny{
            print_constraint_table_row(output, fprinter, ArgInds, ivarp::template get<ArgInds>(rct),
                                       TupleIndexPack<typename RCT::template At<ArgInds>>{})...
        };
    }
}
    template<typename PIType>
    static inline void print_prover_input(std::ostream& output, const PIType& prover_in, FunctionPrinter& fprinter)
    {
        using CTAB = typename PIType::CTArgBounds;
        using RBT = typename PIType::RuntimeBoundTable;
        using RCT = typename PIType::RuntimeConstraintTable;
        impl::print_prover_input_vars<CTAB>(output, prover_in.initial_runtime_bounds, IndexRange<0,PIType::num_args>{},
                                            typename PIType::VariableSplitInfo{}, fprinter);
        impl::print_prover_input_bounds(output, prover_in, fprinter, typename RBT::BoundIndices{});
        impl::print_prover_input_constraints(output, fprinter, prover_in.runtime_constraints, TupleIndexPack<RCT>{});
    }
}
