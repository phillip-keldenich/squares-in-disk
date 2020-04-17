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
// Created by Phillip Keldenich on 03.12.19.
//

#pragma once
#include <iostream>
#include <sstream>
#include <ivarp/run_prover.hpp>
#include <ivarp/critical_printer.hpp>
#include <ivarp/progress_printer.hpp>

struct ProofError : std::logic_error {
    explicit ProofError(const std::string& name) :
        std::logic_error("Error: Proof '" + name + "' returned with critical hypercuboids!")
    {}
};

template<typename ProverInputType, typename ConstraintSystemType, typename Handler> static inline void
    run_proof(const std::string& name, const ProverInputType& p, const ConstraintSystemType& c, const Handler& h)
{
    std::cout << "Starting proof " << name << "..." << std::endl;
    ivarp::ProofInformation info;
    ivarp::ProgressPrinter printer(std::cout, 3, &c);
    ivarp::ProverSettings settings;
    if(!run_prover(p, h, &info, settings, printer)) {
        throw ProofError(name);
    }
    std::cout << "Done: " << info.num_cuboids << " cuboids considered (" << info.num_leaf_cuboids << " leafs)."
              << std::endl;
}
