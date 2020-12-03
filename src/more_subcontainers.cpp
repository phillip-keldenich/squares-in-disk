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
// Created by Phillip Keldenich on 07.07.2020.
//

#include "proof_auxiliaries.hpp"
#include "B4.hpp"
#include "B5.hpp"
#include "B6.hpp"

namespace more_subcontainers_proof1 {
    using namespace aux_functions;
    using namespace B_functions;
    using namespace ivarp;
    using namespace ivarp::args;
    using CTX = DefaultContextWithNumberType<IDouble>;

    // The case for k >= 5, sn <= sigma, y_3 <= 0.
    static const auto s1 = x0;
    static const auto h1 = x1;
    static const auto h2 = x2;
    static const auto h3 = x3;
    static const auto h4 = x4;
    static const auto H4 = 1_Z + T_inv(s1) - h1 - h2 - h3;
    static const auto w1 = w(T_inv(s1), h1);
    static const auto w2 = w(T_inv(s1) - h1, h2);
    static const auto w3 = w(T_inv(s1) - h1 - h2, h3);
    static const auto least_each = T_inv(s1) * (1_Z / 3_Z);

    static const auto F_MSC_1 = square(s1) + 0.83_X * square(sigma(s1)) +
                                B_5(h4, H4) +
                                B_4(T_inv(s1), h1, w1, h2) +
                                B_4(T_inv(s1)-h1, h2, w2, h3) +
                                B_4(T_inv(s1)-h1-h2, h3, w3, h4);

    static const auto F_MSC_1p1 = square(s1) + 0.83_X * square(sigma(s1));
    static const auto F_MSC_1p3 = square(s1) + 0.83_X * square(sigma(s1)) +
                                  B_4(T_inv(s1), h1, w1, h2) +
                                  B_4(T_inv(s1)-h1, h2, w2, h3);

    static const auto system = constraint_system(
        variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X))),
        variable(h1, "h_1", least_each, s1),
        variable(h2, "h_2", least_each, h1),
        variable(h3, "h_3", least_each, h2),
        variable(h4, "h_4", 0_Z, h3),
        0_Z <= H4, H4 <= 1_Z,
        F_MSC_1p1 <= 1.6_X,
        F_MSC_1p3 <= 1.6_X,
        F_MSC_1 <= 1.6_X
    );

    static const auto input = prover_input<
        CTX, U64Pack<
            dynamic_subdivision(128, 4),
            dynamic_subdivision(64, 4),
            dynamic_subdivision(64, 4),
            32, 32>>(system);
}

namespace more_subcontainers_proof2 {
    using namespace aux_functions;
    using namespace B_functions;
    using namespace ivarp;
    using namespace ivarp::args;
    using CTX = DefaultContextWithNumberType<IDouble>;
    static const auto s1 = x0;
    static const auto h1 = x1;
    static const auto h2 = x2;
    static const auto h3 = x3;
    static const auto dely = x4;
    static const auto h_jp1 = x5;

    static const auto w1 = w(T_inv(s1), h1);
    static const auto w2 = w(T_inv(s1) - h1, h2);
    static const auto H_R = maximum(0_Z, T_inv(s1) - h1 - h2) + dely;
    static const auto W_R = w(maximum(0_Z, T_inv(s1) - h1 - h2), H_R);
    static const auto H_jp1 = 1_Z - dely;
    static const auto F_MSC_2 = square(s1) + 0.83_X * square(sigma) +
                                B_4(T_inv(s1), h1, w1, h2) +
                                B_4(T_inv(s1)-h1, h2, w2, h3) +
                                B_5(h_jp1, H_jp1) +
                                B_6(H_R, W_R, h_jp1);
    static const auto F_MSC_2p1 = square(s1) + 0.83_X * square(sigma) +
                                  B_4(T_inv(s1), h1, w1, h2) +
                                  B_4(T_inv(s1)-h1, h2, w2, h3);

    static const auto system = constraint_system(
        variable(s1, "s_1", 0.295_X, sqrt(ensure_expr(1.6_X))),
        variable(h1, "h_1", 0_Z, s1),
        variable(h2, "h_2", 0_Z, h1),
        variable(h3, "h_3", 0_Z, h2),
        variable(dely, "Delta_y", 0_Z, h3),
        variable(h_jp1, "h_{j+1}", 0_Z, h3),
        h1 <= T_inv(s1),
        h2 <= T_inv(s1) - h1,
        h3 <= T_inv(s1) - h1 - h2,
        F_MSC_2 <= 1.6_X,
        F_MSC_2p1 <= 1.6_X
    );

    static const auto input = prover_input<
        CTX, U64Pack<
            dynamic_subdivision(128, 4),
            dynamic_subdivision(64, 4),
            dynamic_subdivision(64, 4),
            32, 32, 16>>(system);

}

static void run_more_subcontainers_proof1() {
    using namespace more_subcontainers_proof1;

    const auto printer = ivarp::critical_printer(std::cerr, more_subcontainers_proof1::system,
                                                 printable_expression("F_MSC_1", F_MSC_1),
                                                 printable_expression("F_MSC_1p1", F_MSC_1p1),
                                                 printable_expression("F_MSC_1p3", F_MSC_1p3));
    run_proof(">= 5 Subcontainers, s_n <= sigma, y_3 <= 0", input,
              more_subcontainers_proof1::system, printer);
}

static void run_more_subcontainers_proof2() {
    using namespace more_subcontainers_proof2;

    const auto printer = ivarp::critical_printer(std::cerr, more_subcontainers_proof2::system,
                                                 printable_expression("F_MSC_2", F_MSC_2),
                                                 printable_expression("F_MSC_2p1", F_MSC_2p1));
    run_proof(">= 5 Subcontainers, s_n <= sigma, y_3 > 0", input,
              more_subcontainers_proof2::system, printer);
}

void run_more_subcontainers() {
    run_more_subcontainers_proof1();
    run_more_subcontainers_proof2();
}

