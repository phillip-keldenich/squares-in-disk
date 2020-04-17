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
// Created by Phillip Keldenich on 14.02.20.
//

#pragma once

namespace ivarp {
    static inline constexpr bool IVARP_HD is_dynamic_subdivision(std::uint64_t subdivision) noexcept {
        return (subdivision >> 32u) != 0;
    }

    constexpr IVARP_HD std::uint64_t dynamic_subdivision(std::uint32_t initial, std::uint32_t subdivision) {
        return (std::uint64_t(initial) << 32u) | subdivision;
    }

    constexpr static inline IVARP_HD std::uint32_t num_initial(std::uint64_t s) noexcept {
        return std::uint32_t(s >> 32u);
    }

    constexpr static inline IVARP_HD std::uint32_t num_subdivision(std::uint64_t s) noexcept {
        return std::uint64_t(~std::uint32_t(0)) & s;
    }

namespace impl {
    template<typename SeqSoFar, std::uint64_t InitialQS, std::uint64_t... Remaining>
        struct ProcessVarSplittingAfterStatic
    {
        struct Type {
            static constexpr std::size_t initial_queue_size = std::size_t(InitialQS);
            using ProcessedSequence = SeqSoFar;
        };
    };

    template<std::uint64_t... S, std::uint64_t InitialQS, std::uint64_t Next, std::uint64_t... Remaining>
        struct ProcessVarSplittingAfterStatic<U64Pack<S...>, InitialQS, Next, Remaining...>
    {
        static_assert(!is_dynamic_subdivision(Next),
                      "Error: Dynamically split variables must be a consecutive prefix of all variables!");
        using Type = typename ProcessVarSplittingAfterStatic<U64Pack<S..., Next>, InitialQS, Remaining...>::Type;
    };

    template<typename SeqSoFar, std::uint64_t InitialQS, std::uint64_t... Remaining>
        struct ProcessVarSplittingBeforeFirstStatic
    {
        struct Type {
            static constexpr std::size_t initial_queue_size = std::size_t(InitialQS);
            using ProcessedSequence = SeqSoFar;
        };
    };

    template<std::uint64_t... S, std::uint64_t InitialQS, std::uint64_t Next, std::uint64_t... Remaining>
        struct ProcessVarSplittingBeforeFirstStatic<U64Pack<S...>, InitialQS, Next, Remaining...>
    {
    private:
        struct NextStatic {
            template<typename = void> struct Lazy :
                ProcessVarSplittingAfterStatic<U64Pack<S..., Next>, InitialQS, Remaining...>
            {};
        };
        struct NextDynamic {
            template<typename = void> struct Lazy :
                ProcessVarSplittingBeforeFirstStatic<U64Pack<S..., Next>, InitialQS * num_initial(Next), Remaining...>
            {};
        };
        using LazyNextType = std::conditional_t<is_dynamic_subdivision(Next), NextDynamic, NextStatic>;
        using NextType = typename LazyNextType::template Lazy<>;

    public:
        using Type = typename NextType::Type;
    };

    template<std::size_t Index, std::uint64_t Split> struct SplitInfo {
        static constexpr std::size_t arg = Index;
        static constexpr std::uint64_t initial = num_initial(Split);
        static constexpr std::uint64_t subdivisions = num_subdivision(Split);
        static constexpr bool is_dynamic = is_dynamic_subdivision(Split);
    };
    template<typename... SplitInfos> struct SplitInfoSequence {};

    template<typename VarSeq, typename SplitSeq> struct MakeSplitInfoSequenceImpl;
    template<std::size_t... VarInds, std::uint64_t... Splits>
    struct MakeSplitInfoSequenceImpl<IndexPack<VarInds...>, U64Pack<Splits...>> {
        using Type = SplitInfoSequence<SplitInfo<VarInds,Splits>...>;
    };
    template<typename VarSeq, typename SplitSeq> using MakeSplitInfoSequence =
        typename MakeSplitInfoSequenceImpl<VarSeq,SplitSeq>::Type;

    template<typename T> struct IsDynamicSplitInfoPredicate {
        static constexpr bool value = T::is_dynamic;
    };
    template<typename T> struct IsStaticSplitInfoPredicate {
        static constexpr bool value = !T::is_dynamic;
    };
    template<typename SplitInfoSeq> struct DynamicSplitInfosImpl;
    template<typename SplitInfoSeq> struct StaticSplitInfosImpl;
    template<typename... SIS> struct DynamicSplitInfosImpl<SplitInfoSequence<SIS...>> {
        using Type = FilterArgsType<SplitInfoSequence, IsDynamicSplitInfoPredicate, SIS...>;
    };
    template<typename... SIS> struct StaticSplitInfosImpl<SplitInfoSequence<SIS...>> {
        using Type = FilterArgsType<SplitInfoSequence, IsStaticSplitInfoPredicate, SIS...>;
    };

    template<typename SplitInfoSeq> using DynamicSplitInfos = typename DynamicSplitInfosImpl<SplitInfoSeq>::Type;
    template<typename SplitInfoSeq> using StaticSplitInfos = typename StaticSplitInfosImpl<SplitInfoSeq>::Type;

    template<typename VarSplitting> struct ProcessVarSplitting;
    template<std::uint64_t First, std::uint64_t... Rest> struct ProcessVarSplitting<U64Pack<First,Rest...>> {
    private:
        static constexpr std::uint64_t ensure_dynamic_first =
            is_dynamic_subdivision(First) ? First : dynamic_subdivision(std::uint32_t(First), 4);
        using Impl = ProcessVarSplittingBeforeFirstStatic<U64Pack<ensure_dynamic_first>,
                                                          num_initial(ensure_dynamic_first), Rest...>;

    public:
        static constexpr std::size_t initial_queue_size = Impl::Type::initial_queue_size;
        using ProcessedSequence = typename Impl::Type::ProcessedSequence;
    };
}
}
