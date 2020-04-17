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

#include "ivarp/symbol_export.hpp"
#include "ivarp/metaprogramming.hpp"
#include <type_traits>
#include <utility>
#include <boost/io/ios_state.hpp>

namespace ivarp {
    namespace impl {
        template<typename BoolType> struct BoolTraitsImpl {
            static IVARP_HD constexpr bool definitely(BoolType b) noexcept {
                return b;
            }

            static IVARP_HD constexpr bool possibly(BoolType b) noexcept {
                return b;
            }

            static IVARP_HD constexpr bool singleton(BoolType b) noexcept {
                return true;
            }
        };

        template<typename BoolType> struct IBoolTraitsImpl {
            static IVARP_HD bool definitely(BoolType b) noexcept {
                return b.definitely();
            }

            static IVARP_HD bool possibly(BoolType b) noexcept {
                return b.possibly();
            }

            static IVARP_HD bool singleton(BoolType b) noexcept {
                return b.singleton();
            }
        };
    }

    template<typename BoolType> struct BoolTraits;
    template<> struct BoolTraits<bool> : impl::BoolTraitsImpl<bool> {};
    template<> struct BoolTraits<const bool> : impl::BoolTraitsImpl<const bool> {};

    class IBool {
    public:
        IBool() = default;
        IVARP_HD IBool(bool lb, bool ub) noexcept : lb(lb), ub(ub) {}
        IVARP_HD IBool(bool b) noexcept : lb(b), ub(b) {} // NOLINT (this is an implicit conversion by design)

        IVARP_HD bool definitely() const noexcept {
            return lb;
        }

        IVARP_HD bool possibly() const noexcept {
            return ub;
        }

        IVARP_HD IBool operator|(IBool other) const noexcept {
            IBool result{*this};
            result |= other;
            return result;
        }

        IVARP_HD IBool &operator|=(IBool other) noexcept {
            lb |= other.lb;
            ub |= other.ub;
            return *this;
        }

        IVARP_HD IBool operator&(IBool other) const noexcept {
            IBool result{*this};
            result &= other;
            return result;
        }

        IVARP_HD IBool &operator&=(IBool other) noexcept {
            lb &= other.lb;
            ub &= other.ub;
            return *this;
        }

        IVARP_HD IBool operator^(IBool other) const noexcept {
            IBool result{*this};
            result ^= other;
            return result;
        }

        IVARP_HD IBool &operator^=(IBool other) noexcept {
            if(lb == ub && other.lb == other.ub) {
                lb = ub = (lb ^ other.lb);
            } else {
                lb = false;
                ub = true;
            }
            return *this;
        }

        IVARP_HD IBool operator!() const noexcept {
            return {!ub, !lb};
        }

        IVARP_HD IBool join(IBool other) const noexcept {
            IBool result{*this};
            result.do_join(other);
            return result;
        }

        IVARP_HD void do_join(IBool other) noexcept {
            lb &= other.lb;
            ub |= other.ub;
        }

        IVARP_HD bool same(IBool other) const noexcept {
            return lb == other.lb && ub == other.ub;
        }

        IVARP_HD bool singleton() const noexcept {
            return lb == ub;
        }

    private:
        bool lb, ub;
    };

    static inline std::ostream& operator<<(std::ostream& o, IBool i) {
        boost::io::ios_flags_saver saver(o);
        o << '[' << std::boolalpha << i.definitely() << ", " << i.possibly() << ']';
        return o;
    }

    template<typename T> using IsBoolean = std::integral_constant<bool,
            std::is_same<BareType<T>,bool>::value || std::is_same<BareType<T>, IBool>::value
        >;

    template<> struct BoolTraits<IBool> : impl::IBoolTraitsImpl<IBool> {};
    template<> struct BoolTraits<const IBool> : impl::IBoolTraitsImpl<const IBool> {};

    template<typename BoolType> static inline IVARP_HD bool definitely(const BoolType& b) noexcept {
        return BoolTraits<BareType<BoolType>>::definitely(b);
    }

    template<typename BoolType> static inline IVARP_HD bool possibly(const BoolType& b) noexcept {
        return BoolTraits<BareType<BoolType>>::possibly(b);
    }

    template<typename BoolType> static inline IVARP_HD bool singleton(const BoolType& b) noexcept {
        return BoolTraits<BareType<BoolType>>::singleton(b);
    }
}
