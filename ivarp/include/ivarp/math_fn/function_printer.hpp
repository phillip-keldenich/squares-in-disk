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
// Created by Phillip Keldenich on 17.02.20.
//

#pragma once

namespace ivarp {
    enum class BoundPrintOption {
        ALL,
        PARENT_UNBOUNDED,
        NONE
    };

    enum class FoldedPrintOption {
        MARKED_WITH_CHILDREN_DOUBLE,
        MARKED_WITH_CHILDREN_RATIONAL,
        CHILDREN,
        AS_RATIONAL_CONSTANT,
        AS_DOUBLE_CONSTANT
    };

    struct PrintOptions {
        BoundPrintOption print_bounds = BoundPrintOption::PARENT_UNBOUNDED;

        /// How to handle folded constants
        FoldedPrintOption print_folded = FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE;
    };

namespace impl {
    template<typename Tag, typename = void> struct HasPrintOp : std::false_type {};
    template<typename Tag> struct HasPrintOp<Tag, MakeVoid<decltype(Tag::print_operator)>> : std::true_type {};
    template<typename Tag, typename = void> struct HasTagName : std::false_type {};
    template<typename Tag> struct HasTagName<Tag, MakeVoid<decltype(Tag::name())>> : std::true_type {};

    template<typename Tag, std::enable_if_t<HasPrintOp<Tag>::value, int> = 0>
        static inline PrintOp get_print_operator(Tag)
    {
        return Tag::print_operator;
    }

    template<typename Tag, std::enable_if_t<!HasPrintOp<Tag>::value, int> = 0>
        static inline PrintOp get_print_operator(Tag)
    {
        return PrintOp::NONE;
    }

    template<typename Tag, std::enable_if_t<HasTagName<Tag>::value, int> = 0>
        static inline const char* get_tag_name(Tag)
    {
        return Tag::name();
    }

    template<typename Tag, std::enable_if_t<!HasTagName<Tag>::value, int> = 0>
        static inline const char* get_tag_name(Tag t)
    {
        return typeid(t).name();
    }

    static inline const char* get_operator_str(PrintOp op) noexcept {
        switch(op) {
            case PrintOp::DIV: return "/";
            case PrintOp::MINUS: return "-";
            default: return "?????";
            case PrintOp::AND: return "&&";
            case PrintOp::OR: return "||";
            case PrintOp::EQ: return "==";
            case PrintOp::XOR: return "^";
            case PrintOp::LOG_NEG: return "!";
            case PrintOp::MUL: return "*";
            case PrintOp::PLUS: return "+";
            case PrintOp::UNARY_MINUS: return "-";
            case PrintOp::LEQ: return "<=";
            case PrintOp::GEQ: return ">=";
            case PrintOp::GT: return ">";
            case PrintOp::LT: return "<";
            case PrintOp::NEQ: return "!=";
            case PrintOp::TILDE: return "~";
        }
    }

    template<typename MathExprOrPred> static constexpr inline bool printing_is_actually_bounded(const MathExprOrPred&) {
        return false;
    }
    template<typename Child, typename BoundType> static constexpr inline bool
        printing_is_actually_bounded(const BoundedMathExpr<Child,BoundType>&)
    {
        return fixed_point_bounds::is_lb(BoundType::lb) || fixed_point_bounds::is_ub(BoundType::ub);
    }
    template<typename Child, typename BoundType> static constexpr inline bool
        printing_is_actually_bounded(const BoundedPredicate<Child,BoundType>&)
    {
        return BoundType::lb || !BoundType::ub;
    }

    template<typename BoundedType>
        static inline PrintRank bounded_get_print_rank(const BoundedType& b, bool parent_bounded,
                                                       const PrintOptions& opts);

    template<typename Child, std::int64_t LB, std::int64_t UB>
        static inline PrintRank get_print_rank(const ConstantFoldedExpr<Child,LB,UB>& cf, bool parent_bounded,
                                               const PrintOptions& opts)
    {
        switch(opts.print_folded) {
            case FoldedPrintOption::MARKED_WITH_CHILDREN_RATIONAL:
            case FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE:
                return PrintRank::PARENTHESIS;
            case FoldedPrintOption::AS_DOUBLE_CONSTANT:
                return PrintRank::ID;
            case FoldedPrintOption::AS_RATIONAL_CONSTANT:
                return PrintRank::ID;
            case FoldedPrintOption::CHILDREN:
                return get_print_rank(cf.base, parent_bounded, opts);
        }
    }

    template<typename Child, bool LB, bool UB>
        static inline PrintRank get_print_rank(const ConstantFoldedPred<Child,LB,UB>& cf, bool parent_bounded,
                                               const PrintOptions& opts)
    {
        switch(opts.print_folded) {
            case FoldedPrintOption::MARKED_WITH_CHILDREN_RATIONAL:
            case FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE:
                return PrintRank::PARENTHESIS;
            case FoldedPrintOption::AS_DOUBLE_CONSTANT:
                return PrintRank::ID;
            case FoldedPrintOption::AS_RATIONAL_CONSTANT:
                return PrintRank::ID;
            case FoldedPrintOption::CHILDREN:
                return get_print_rank(cf.base, parent_bounded, opts);
        }
    }

    template<typename BoundType, typename Child>
    static inline PrintRank get_print_rank(const BoundedMathExpr<Child,BoundType>& b,
                                           bool parent_bounded, const PrintOptions& opts)
    {
        return bounded_get_print_rank(b, parent_bounded, opts);
    }

    template<typename BoundType, typename Child>
    static inline PrintRank get_print_rank(const BoundedPredicate<Child,BoundType>& b,
                                           bool parent_bounded, const PrintOptions& opts)
    {
        return bounded_get_print_rank(b, parent_bounded, opts);
    }

    template<typename IndexType>
    static inline PrintRank get_print_rank(const MathArg<IndexType>&, bool parent_bounded, const PrintOptions&)
    {
        return PrintRank::ID;
    }

    template<typename T, std::int64_t LB, std::int64_t UB>
    static inline PrintRank get_print_rank(const MathConstant<T,LB,UB>&, bool parent_bounded, const PrintOptions&)
    {
        return PrintRank::ID;
    }

    template<std::int64_t LB, std::int64_t UB>
    static inline PrintRank get_print_rank(const MathCUDAConstant<LB,UB>&, bool parent_bounded, const PrintOptions&)
    {
        return PrintRank::ID;
    }

    template<typename T, bool LB, bool UB>
    static inline PrintRank get_print_rank(const MathBoolConstant<T,LB,UB>&, bool parent_bounded, const PrintOptions&)
    {
        return PrintRank::ID;
    }

    template<typename Tag, typename Arg>
    static inline PrintRank get_print_rank(const MathUnary<Tag, Arg>&, bool parent_bounded,
                                           const PrintOptions&)
    {
        return get_print_operator(Tag{}) == PrintOp::NONE ? PrintRank::PARENTHESIS : PrintRank::UNARY;
    }

    template<typename Tag, typename Arg>
    static inline PrintRank get_print_rank(const UnaryMathPred<Tag,Arg>&, bool parent_bounded,
                                           const PrintOptions&)
    {
        return PrintRank::UNARY;
    }

    template<typename Tag, typename Arg1, typename Arg2>
    static inline PrintRank get_print_rank(const MathBinary<Tag,Arg1,Arg2>&, bool parent_bounded, const PrintOptions&)
    {
        switch(get_print_operator(Tag{})) {
            case PrintOp::MINUS:
            case PrintOp::PLUS:
                return PrintRank::ADDITIVE;

            case PrintOp::MUL:
            case PrintOp::DIV:
                return PrintRank::MULTIPLICATIVE;

            default:
                return PrintRank::UNKNOWN;
        }
    }

    template<typename Tag, typename Arg1, typename Arg2>
    static inline PrintRank get_print_rank(const BinaryMathPred<Tag,Arg1,Arg2>&, bool parent_bounded, const PrintOptions&)
    {
       switch(get_print_operator(Tag{})) {
           case PrintOp::XOR:
               return PrintRank::XOR;
           default:
               return PrintRank::COMPARISON;
       }
    }

    template<typename Tag, typename Arg1, typename Arg2, typename Arg3>
    static inline PrintRank get_print_rank(const MathTernary<Tag,Arg1,Arg2,Arg3>&, bool parent_bounded, const PrintOptions&)
    {
        return PrintRank::PARENTHESIS;
    }

    template<typename Tag, typename... Args>
    static inline PrintRank get_print_rank(const MathNAry<Tag,Args...>&, bool parent_bounded, const PrintOptions&) {
        return PrintRank::PARENTHESIS;
    }

    template<typename Tag, typename... Args>
    static inline PrintRank get_print_rank(const NAryMathPred<Tag,Args...>& m, bool parent_bounded, const PrintOptions&) {
        switch(get_print_operator(Tag{})) {
            case PrintOp::AND:
                return PrintRank::AND;

            case PrintOp::OR:
                return PrintRank::OR;

            default:
                return PrintRank::UNKNOWN;
        }
    }

    template<typename F, typename... Args>
    static inline PrintRank get_print_rank(const MathCustomFunction<F,Args...>&, bool parent_bounded, const PrintOptions&) {
        return PrintRank::PARENTHESIS;
    }

    template<typename F, typename... Args>
    static inline PrintRank get_print_rank(const MathCustomPredicate<F,Args...>&, bool parent_bounded, const PrintOptions&) {
        return PrintRank::PARENTHESIS;
    }

    template<typename BoundedType>
        static inline PrintRank bounded_get_print_rank(const BoundedType& b, bool parent_bounded,
                                                       const PrintOptions& opts)
    {
        if(!printing_is_actually_bounded(b)) {
            return get_print_rank(b.child, parent_bounded, opts);
        }

        if(opts.print_bounds == BoundPrintOption::ALL) {
            return PrintRank::PARENTHESIS;
        } else if(opts.print_bounds == BoundPrintOption::NONE) {
            return get_print_rank(b.child, true, opts);
        } else {
            if(!parent_bounded) {
                return PrintRank::PARENTHESIS;
            } else {
                return get_print_rank(b.child, true, opts);
            }
        }
    }
}

    class DefaultArgNameLookup : public ArgNameLookup {
    public:
        IVARP_H std::string arg_name(std::size_t arg_index) const override {
            char buf[32];
            int n = std::snprintf(buf, 32, "x%llu", static_cast<unsigned long long>(arg_index));
            return std::string(buf, std::size_t(n));
        }
    };

    class FunctionPrinter {
    public:
        static IVARP_EXPORTED_SYMBOL FunctionPrinter& get_default_printer();

        explicit FunctionPrinter(const PrintOptions& options, const ArgNameLookup* arg_names) :
            arg_names(arg_names),
            options(options)
        {}

        const ArgNameLookup *get_arg_name_lookup() const noexcept {
            return arg_names;
        }

        template<typename MathExprOrPred>
            void print(std::ostream& output, const MathExprOrPred& p)
        {
            do_print(output, p, false);
        }

        template<typename CustomMathExprOrPred, typename = MakeVoid<typename CustomMathExprOrPred::FunctorType>>
            void register_custom_name(const CustomMathExprOrPred& p, std::string name)
        {
            std::string s{typeid(p.functor).name()};
            custom_function_names[s] = ivarp::move(name);
        }

        void print_arg(std::ostream& output, std::size_t index) const {
            output << arg_names->arg_name(index);
        }

    private:
        template<typename Bounds, std::enable_if_t<std::is_same<BareType<decltype(Bounds::lb)>,bool>::value,int> = 0>
            void do_print_bounds(std::ostream& output)
        {
            output << std::boolalpha << '[' << Bounds::lb << ", " << Bounds::ub << "]";
        }

        template<typename Bounds, std::enable_if_t<std::is_same<BareType<decltype(Bounds::lb)>,std::int64_t>::value,int> = 0>
            void do_print_bounds(std::ostream& output)
        {
            output << '[' << fixed_point_bounds::PrintFixedPoint(Bounds::lb) << ", " <<
                             fixed_point_bounds::PrintFixedPoint(Bounds::ub) << ']';
        }

        void do_print_args(std::ostream& output, bool parent_bounded)
        {
            output << "()";
        }

        template<typename A1, typename... Args>
            void do_print_args(std::ostream& output, bool parent_bounded, const A1& a1, const Args&... args)
        {
            output << '(';
            do_print(output, a1, parent_bounded);
            ConstructWithAny{(do_print(output << ", ", args, parent_bounded),0)...};
            output << ')';
        }

        template<typename Child, typename Bounds>
            void do_print_bounded(std::ostream& output, const BoundedMathExpr<Child,Bounds>& b)
        {
            output << "{ ";
            do_print(output, b.child, true);
            output << " }";
            do_print_bounds<Bounds>(output);
        }

        template<typename Child, typename Bounds>
            void do_print_bounded(std::ostream& output, const BoundedPredicate<Child,Bounds>& b)
        {
            output << "{ ";
            do_print(output, b.child, true);
            output << " }";
            do_print_bounds<Bounds>(output);
        }

        template<typename Child, typename Bounds>
            void do_print(std::ostream& output, const BoundedMathExpr<Child,Bounds>& b, bool parent_bounded)
        {
            if(!impl::printing_is_actually_bounded(b)) {
                do_print(output, b.child, parent_bounded);
                return;
            }

            if(options.print_bounds == BoundPrintOption::ALL ||
              (options.print_bounds == BoundPrintOption::PARENT_UNBOUNDED && !parent_bounded))
            {
                do_print_bounded(output, b);
                return;
            }

            do_print(output, b.child, true);
        }

        template<typename Child, typename Bounds>
            void do_print(std::ostream& output, const BoundedPredicate<Child,Bounds>& b, bool parent_bounded)
        {
            if(!impl::printing_is_actually_bounded(b)) {
                do_print(output, b.child, parent_bounded);
                return;
            }

            if(options.print_bounds == BoundPrintOption::ALL ||
              (options.print_bounds == BoundPrintOption::PARENT_UNBOUNDED && !parent_bounded))
            {
                do_print_bounded(output, b);
                return;
            }

            do_print(output, b.child, true);
        }

        template<typename IndexType>
            void do_print(std::ostream& output, const MathArg<IndexType>&, bool)
        {
            output << arg_names->arg_name(IndexType::value);
        }

        void unary_print_prefix(std::ostream& output, const MathOperatorTagUnaryMinus&) {
            output << '-';
        }

        void unary_print_prefix(std::ostream& output, const UnaryMathPredNotTag&) {
            output << '!';
        }

        void unary_print_prefix(std::ostream& output, const UnaryMathPredUndefTag&) {
            output << '~';
        }

        template<typename Tag>
        void unary_print_prefix(std::ostream& output, const Tag&)
        {
            output << Tag::name();
        }

        template<typename Tag, typename Arg>
            void do_print(std::ostream& output, const MathUnary<Tag,Arg>& u, bool parent_bounded)
        {
            bool paren = (impl::get_print_rank(u.arg, parent_bounded, options) > PrintRank::UNARY) ||
                         impl::get_print_operator(Tag{}) == PrintOp::NONE;
            unary_print_prefix(output, Tag{});
            do_print_paren(output, u.arg, parent_bounded, paren);
        }

        template<unsigned Pow, typename Arg>
            void do_print(std::ostream& output, const MathUnary<MathFixedPowTag<Pow>,Arg>& p, bool parent_bounded)
        {
            bool paren = (impl::get_print_rank(p.arg, parent_bounded, options) > PrintRank::PARENTHESIS);
            do_print_paren(output, p.arg, parent_bounded, paren);
            output << "**" << Pow;
        }

        template<typename MathExprOrPred>
            void do_print_paren(std::ostream& out, const MathExprOrPred& m, bool parent_bounded, bool paren)
        {
            if(paren) {
                out << '(';
            }
            do_print(out, m, parent_bounded);
            if(paren) {
                out << ')';
            }
        }

        template<typename Tag, typename MathExprOrPred>
            void do_print_binary(std::ostream& output, const MathExprOrPred& b, bool parent_bounded)
        {
            PrintOp self_op = impl::get_print_operator(Tag{});
            if(self_op == PrintOp::NONE) {
                const char* n = impl::get_tag_name(Tag{});
                output << n << '(';
                do_print(output, b.arg1, parent_bounded);
                output << ", ";
                do_print(output, b.arg2, parent_bounded);
                output << ')';
            } else {
                PrintRank self_rank = impl::get_print_rank(b, parent_bounded, options);
                PrintRank r1 = impl::get_print_rank(b.arg1, parent_bounded, options);
                PrintRank r2 = impl::get_print_rank(b.arg2, parent_bounded, options);
                bool paren1 = (self_rank < r1);
                bool paren2 = (self_rank < r2);
                paren2 |= (self_rank == r2 && self_op != PrintOp::PLUS &&
                           self_op != PrintOp::MUL && self_op != PrintOp::XOR);
                do_print_paren(output, b.arg1, parent_bounded, paren1);
                output << ' ';
                output << impl::get_operator_str(self_op);
                output << ' ';
                do_print_paren(output, b.arg2, parent_bounded, paren2);
            }
        }

        template<typename Tag, typename Arg1, typename Arg2>
            void do_print(std::ostream& output, const MathBinary<Tag,Arg1,Arg2>& b, bool parent_bounded)
        {
            do_print_binary<Tag>(output, b, parent_bounded);
        }

        template<typename Tag, typename Arg1, typename Arg2>
            void do_print(std::ostream& output, const BinaryMathPred<Tag,Arg1,Arg2>& b, bool parent_bounded)
        {
            do_print_binary<Tag>(output, b, parent_bounded);
        }

        template<typename Tag, typename Arg1, typename Arg2, typename Arg3>
            void do_print(std::ostream& output, const MathTernary<Tag, Arg1, Arg2, Arg3>& b, bool parent_bounded)
        {
            output << impl::get_tag_name(Tag{}) << '(';
            do_print(output, b.arg1, parent_bounded);
            output << ", ";
            do_print(output, b.arg2, parent_bounded);
            output << ", ";
            do_print(output, b.arg3, parent_bounded);
            output << ')';
        }

        void do_print_constant(std::ostream& o, const Rational& r) const {
            if (r.get_den() != 1) {
                o << '(' << r << ')';
            } else {
                o << r.get_num();
            }
        }

        template<typename T, std::enable_if_t<IsIntervalType<T>::value, int> = 0>
            void do_print_constant(std::ostream& o, const T& t)
        {
            if (t.singleton()) {
                o << t.lb();
            } else {
                o << t;
            }
        }

        template<typename T, std::enable_if_t<!IsRational<T>::value && !IsIntervalType<T>::value, int> = 0>
            void do_print_constant(std::ostream& o, const T& t)
        {
            o << t;
        }

        template<typename T, std::int64_t LB, std::int64_t UB>
            void do_print(std::ostream& output, const MathConstant<T, LB, UB>& c, bool)
        {
            do_print_constant(output, c.value);
        }

        template<std::int64_t LB, std::int64_t UB>
        void do_print(std::ostream& output, const MathCUDAConstant<LB, UB>& c, bool)
        {
            do_print_constant(output, c.idouble);
        }

        template<typename T, bool LB, bool UB>
        void do_print(std::ostream& output, const MathBoolConstant<T, LB, UB>& c, bool) {
            if (possibly(c.value) == definitely(c.value)) {
                output << possibly(c.value);
            } else {
                output << c.value;
            }
        }

        template<typename Child, std::int64_t LB, std::int64_t UB>
            void do_print(std::ostream& output, const ConstantFoldedExpr<Child, LB, UB>& c, bool parent_bounded)
        {
            switch(options.print_folded) {
                case FoldedPrintOption::AS_DOUBLE_CONSTANT:
                    do_print_constant(output, c.idouble);
                    return;
                case FoldedPrintOption::AS_RATIONAL_CONSTANT:
                    do_print_constant(output, c.irational);
                    return;
                case FoldedPrintOption::CHILDREN:
                    do_print(output, c.base, parent_bounded);
                    return;
                case FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE:
                case FoldedPrintOption::MARKED_WITH_CHILDREN_RATIONAL:
                    output << "folded{";
                    do_print(output, c.base, parent_bounded);
                    output << " : ";
                    if(options.print_folded == FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE) {
                        do_print_constant(output, c.idouble);
                    } else {
                        do_print_constant(output, c.irational);
                    }
                    output << '}';
                    return;
            }
        }

        template<typename Child, bool LB, bool UB>
            void do_print(std::ostream& output, const ConstantFoldedPred<Child, LB, UB>& c, bool parent_bounded)
        {
            switch(options.print_folded) {
                case FoldedPrintOption::AS_DOUBLE_CONSTANT:
                case FoldedPrintOption::AS_RATIONAL_CONSTANT:
                    if (possibly(c.value) == definitely(c.value)) {
                        output << possibly(c.value);
                    } else {
                        output << c.value;
                    }
                    return;
                case FoldedPrintOption::CHILDREN:
                    do_print(output, c.base, parent_bounded);
                    return;
                case FoldedPrintOption::MARKED_WITH_CHILDREN_DOUBLE:
                case FoldedPrintOption::MARKED_WITH_CHILDREN_RATIONAL:
                    output << "folded{";
                    do_print(output, c.base, parent_bounded);
                    output << " : ";
                    if (possibly(c.value) == definitely(c.value)) {
                        output << possibly(c.value);
                    } else {
                        output << c.value;
                    }
                    output << '}';
                    return;
            }
        }

        template<typename Tag, typename NA, std::size_t... Indices>
            void do_print_nary(std::ostream& output, const NA& n, bool parent_bounded, IndexPack<Indices...>)
        {
            output << impl::get_tag_name(Tag{});
            do_print_args(output, parent_bounded, ivarp::template get<Indices>(n.args)...);
        }

        template<typename Element>
            int do_print_logic_element(std::ostream& output, bool parent_bounded, const Element& e)
        {
            bool paren = (impl::get_print_rank(e, parent_bounded, options) > PrintRank::COMPARISON);
            do_print_paren(output, e, parent_bounded, paren);
            return 0;
        }

        template<typename Tag, typename NA, std::size_t I1, std::size_t... Indices>
            void do_print_nary_logic(std::ostream& output, bool parent_bounded, const NA& n, PrintOp op, IndexPack<I1, Indices...>)
        {
            const char* cop = (op == PrintOp::AND ? " && " : " || ");
            do_print_logic_element(output, parent_bounded, ivarp::template get<I1>(n.args));
            ConstructWithAny{do_print_logic_element(output << cop, parent_bounded, ivarp::template get<Indices>(n.args))...};
        }

        template<typename Tag, typename... Args>
            void do_print(std::ostream& output, const MathNAry<Tag,Args...>& n, bool parent_bounded)
        {
            do_print_nary<Tag>(output, n, parent_bounded, IndexRange<0,sizeof...(Args)>{});
        }

        template<typename Tag, typename... Args>
            void do_print(std::ostream& output, const NAryMathPred<Tag,Args...>& n, bool parent_bounded)
        {
            PrintOp op = impl::get_print_operator(Tag{});
            if(op == PrintOp::NONE) {
                do_print_nary<Tag>(output, n, parent_bounded, IndexRange<0,sizeof...(Args)>{});
            } else {
                do_print_nary_logic<Tag>(output, n, parent_bounded, op, IndexRange<0,sizeof...(Args)>{});
            }
        }

        template<typename C, std::size_t... Indices>
            void do_print_custom(std::ostream& output, bool parent_bounded, const C& c, IndexPack<Indices...>)
        {
            do_print_args(output, parent_bounded, ivarp::template get<Indices>(c.args)...);
        }

        template<typename FnType, typename... Args>
            void do_print(std::ostream& output, const MathCustomFunction<FnType, Args...>& n, bool parent_bounded)
        {
            output << get_custom_name(n.functor);
            do_print_custom(output, parent_bounded, n, IndexRange<0,sizeof...(Args)>{});
        }

        template<typename FnType, typename... Args>
            void do_print(std::ostream& output, const MathCustomPredicate<FnType, Args...>& n, bool parent_bounded)
        {
            output << get_custom_name(n.functor);
            do_print_custom(output, parent_bounded, n, IndexRange<0,sizeof...(Args)>{});
        }

        template<typename FType>
            const std::string& get_custom_name(const FType& ft)
        {
            std::string t{typeid(ft).name()};
            auto iter = custom_function_names.find(t);
            if(iter == custom_function_names.end()) {
                std::stringstream o;
                o << "custom" << custom_function_names.size();
                iter = custom_function_names.emplace(t, o.str()).first;
            }
            return iter->second;
        }

        const ArgNameLookup* arg_names;
        std::unordered_map<std::string, std::string> custom_function_names;
        PrintOptions options;
    };

    template<typename MathExprOrPred, std::enable_if_t<IsMathExprOrPred<MathExprOrPred>::value, int> = 0>
        static inline std::ostream& operator<<(std::ostream& o, const MathExprOrPred& p)
    {
        FunctionPrinter::get_default_printer().print(o, p);
        return o;
    }
}
