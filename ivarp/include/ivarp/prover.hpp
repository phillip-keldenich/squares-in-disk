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
// Created by Phillip Keldenich on 29.10.19.
//

#pragma once

#include <atomic>
#include <vector>
#include <sstream>
#include <iostream>
#include <string>
#include <thread>

#include "ivarp/number.hpp"
#include "ivarp/math_fn.hpp"
#include "ivarp/prover/proof_information.hpp"
#include "ivarp/constant_folding.hpp"
#include "ivarp/compile_time_bounds.hpp"
#include "ivarp/splitter.hpp"
#include "ivarp/default_progress_observer.hpp"

#include "prover/cuboid_queue.hpp"
#include "prover/variable.hpp"
#include "prover/constraints.hpp"
#include "prover/check_predicate_to_constraints.hpp"
#include "prover/dynamic_entry.hpp"
#include "prover/prover.hpp"
#include "prover/constraint_propagation.hpp"
#include "prover/proof_runner.hpp"
#include "prover/prover_cpu.hpp"
#include "prover/prover_thread.hpp"
#include "prover/proof_runner_impl.hpp"
