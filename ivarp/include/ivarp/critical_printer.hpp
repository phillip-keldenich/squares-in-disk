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
// Created by me on 26.11.19.
//

#include "ivarp/math_fn.hpp"

namespace ivarp {
    /// A critical handler that prints all variables of the given criticals and optionally, a list of further expressions evaluated
    /// on the critical hypercuboids (use printable_expression for that purpose).
    /// The locking parameter controls whether the actual printing takes place under a std::mutex lock; if we are printing to
    /// std::cout or std::cerr, locking is typically not necessary.
    template<bool Locking, typename... OutputExprs> class CriticalPrinter {
    public:
        template<typename ConstraintSystemType, typename... OE>
        explicit CriticalPrinter(const ConstraintSystemType* lookup, std::ostream* output, OE&&... exprs) :
            name_lookup(lookup),
            num_args(ConstraintSystemType::num_args),
            output(output),
            exprs(std::forward<OE>(exprs)...)
        {}

        template<typename Context, typename ValueArray> void operator()(const Context&, const ValueArray& values) const {
            std::ostringstream out;
            out<< "Critical:\n";
            for(std::size_t i = 0; i < num_args; ++i) {
                out<< '\t' << name_lookup->arg_name(i) << ": " << values[i] << std::endl;
            }
            output_exprs<Context>(out, values, IndexRange<0,sizeof...(OutputExprs)>{});
            do_print(out);
        }

    private:
        void do_print(const std::ostringstream& o) const {
            LockGuard guard{lock};
            *output << o.str();
        }

        template<typename Context, typename ArgArray, std::size_t I1, std::size_t... Inds>
            void output_exprs(std::ostringstream& out, const ArgArray& args, IndexPack<I1,Inds...>) const
        {
            out << '\t' << get<I1>(exprs).name << ": "
                << get<I1>(exprs).expr.template array_evaluate<Context>(args) << std::endl;
            output_exprs<Context>(out, args, IndexPack<Inds...>{});
        }

        template<typename Context, typename ArgArray>
            void output_exprs(std::ostringstream&, const ArgArray&, IndexPack<>) const
        {}

        struct NoOpLock {};
        struct NoOpGuard {
            explicit NoOpGuard(NoOpLock&) noexcept {}
        };

        using Lock = std::conditional_t<Locking, std::mutex, NoOpLock>;
        using LockGuard = std::conditional_t<Locking, std::unique_lock<std::mutex>, NoOpGuard>;

        mutable Lock lock;
        const ArgNameLookup* name_lookup;
        std::size_t num_args;
        std::ostream* output;
        std::tuple<OutputExprs...> exprs;
    };

    template<typename ExpressionType> struct PrintExpr {
        template<typename StringArg, typename EType> explicit PrintExpr(StringArg&& arg, EType&& expr) :
            name(std::forward<StringArg>(arg)), expr(std::forward<EType>(expr))
        {}

        std::string name;
        ExpressionType expr;
    };

    template<typename StringArg, typename EType> static inline auto printable_expression(StringArg&& name, EType&& expr) {
        return PrintExpr<std::decay_t<EType>>{std::forward<StringArg>(name), std::forward<EType>(expr)};
    }

    template<typename ConstraintSystemType, typename... PrintExprs>
        static inline auto critical_printer(std::ostream& output, const ConstraintSystemType& p, PrintExprs&&... exprs)
    {
        return CriticalPrinter<false, std::decay_t<PrintExprs>...>{&p, &output, std::forward<PrintExprs>(exprs)...};
    }

    template<typename ConstraintSystemType, typename... PrintExprs> static inline auto
        locked_critical_printer(std::ostream& output, const ConstraintSystemType& p, PrintExprs&&... exprs)
    {
        return CriticalPrinter<true, std::decay_t<PrintExprs>...>{&p, &output, std::forward<PrintExprs>(exprs)...};
    }
}
