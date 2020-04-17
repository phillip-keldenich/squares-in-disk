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
// Created by Phillip Keldenich on 28.02.20.
//

#include "ivarp/number.hpp"
#include <vector>

namespace ivarp {
    static inline unsigned pull_out_factor(unsigned rem, unsigned fac, std::vector<FactorEntry>& result) {
        unsigned m = 0;
        while(rem % fac == 0) {
            rem /= fac;
            ++m;
        }
        if(m) {
            result.push_back(FactorEntry{fac, m});
        }
        return rem;
    }

    static inline unsigned pull_out_small(unsigned rem, std::vector<FactorEntry>& result) {
        const unsigned small_facs[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
        for(unsigned fac : small_facs) {
            rem = pull_out_factor(rem, fac, result);
            if(rem == 1) {
                break;
            }
        }
        return rem;
    }

    std::vector<FactorEntry> factorize(unsigned number) {
        std::vector<FactorEntry> result;
        constexpr unsigned maxvalue = (1u << (sizeof(unsigned) * CHAR_BIT / 2u)) - 1;
        constexpr unsigned maxcurr = maxvalue - maxvalue % 30;
        const unsigned large_facs[] = {1, 7, 11, 13, 17, 19, 23, 29};

        unsigned rem = pull_out_small(number, result);
        if(rem == 1) {
            return result;
        }
        if(rem < 961) {
            result.push_back(FactorEntry{rem, 1});
            return result;
        }

        for(unsigned curr = 30; curr <= maxcurr && curr*curr < rem; curr += 30) {
            for(unsigned fac_offs : large_facs) {
                unsigned fac = curr + fac_offs;
                rem = pull_out_factor(rem, fac, result);
            }
        }

        if(rem != 1) {
            result.push_back(FactorEntry{rem, 1});
        }
        return result;
    }

    namespace {
        struct FactorPackingInfo {
            explicit FactorPackingInfo(unsigned input) :
                factorization(factorize(input)),
                current_product(1),
                max_product(input)
            {}

            FactorPackingInfo() = default;

            void advance(unsigned factor) noexcept {
                while(!factorization.empty() && factorization.back().factor > factor) {
                    factorization.pop_back();
                }
            }

            bool fits(unsigned factor) const noexcept {
                if(factorization.empty()) {
                    return false;
                }
                if(current_product * factor > max_product) {
                    return false;
                }
                return factorization.back().factor == factor;
            }

            bool gap(unsigned factor) const noexcept {
                return current_product * factor <= max_product;
            }

            void pack(unsigned factor) noexcept {
                current_product *= factor;
                if(!factorization.empty() && factorization.back().factor == factor) {
                    if(--factorization.back().multiplicity == 0) {
                        factorization.pop_back();
                    }
                }
            }

            std::vector<FactorEntry> factorization;
            unsigned current_product, max_product;
        };
    }

    void pack_factors(unsigned packed_num, unsigned num_buckets, const unsigned *bucket_in, unsigned* bucket_out) {
        std::vector<FactorEntry> packed_fac = factorize(packed_num);
        std::reverse(packed_fac.begin(), packed_fac.end());
        std::unique_ptr<FactorPackingInfo[]> state =
            std::make_unique<FactorPackingInfo[]>(num_buckets);

        for(unsigned i = 0; i < num_buckets; ++i) {
            state[i] = FactorPackingInfo{i};
        }

        for(const auto& factor : packed_fac) {
            for(std::size_t i = 0; i < num_buckets; ++i) {
                state[i].advance(factor.factor);
            }

            for(unsigned j = 0; j < factor.multiplicity; ++j) {
                std::size_t index_fitting = ~std::size_t(0);
                unsigned smallest_fitting = std::numeric_limits<unsigned>::max();
                std::size_t index_gap = ~std::size_t(0);
                unsigned smallest_gap = std::numeric_limits<unsigned>::max();
                std::size_t index_smallest = ~std::size_t(0);
                unsigned smallest = std::numeric_limits<unsigned>::max();

                for(std::size_t i = 0; i < num_buckets; ++i) {
                    unsigned curr = state[i].current_product;
                    if(state[i].fits(factor.factor) && curr < smallest_fitting) {
                        index_fitting = i;
                        smallest_fitting = curr;
                    }
                    if(state[i].gap(factor.factor) && curr < smallest_gap) {
                        index_gap = i;
                        smallest_gap = curr;
                    }
                    if(curr < smallest) {
                        index_smallest = i;
                        smallest = curr;
                    }
                }

                std::size_t index = (index_fitting < num_buckets) ? index_fitting :
                                    (index_gap < num_buckets) ? index_gap :  index_smallest;
                state[index].pack(factor.factor);
            }
        }

        for(std::size_t i = 0; i < num_buckets; ++i) {
            bucket_out[i] = std::min(state[i].current_product, state[i].max_product);
        }
    }
}
