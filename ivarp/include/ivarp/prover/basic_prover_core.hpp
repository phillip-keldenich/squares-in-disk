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
// Created by Phillip Keldenich on 25.02.20.
//

#pragma once

namespace ivarp {
    template<typename ProverInputType, typename OnCritical> class BasicProverCore {
    private:
        using Context = typename ProverInputType::Context;
        using RBT = typename ProverInputType::RuntimeBoundTable;
        using RCT = typename ProverInputType::RuntimeConstraintTable;
        using DBA = DynamicBoundApplication<RBT, Context>;

    public:
        virtual void set_settings(const ProverSettings* settings) {
            this->settings = settings;
        }

        void set_runtime_bounds(const RBT* rbt) noexcept {
            this->rbt = rbt;
        }

        void set_runtime_constraints(const RCT* rct) noexcept {
            this->rct = rct;
        }

        void set_dynamic_bound_application(const DBA* dba) noexcept {
            this->dba = dba;
        }

        void set_on_critical(const OnCritical* on_critical) noexcept {
            this->on_critical = on_critical;
        }

    protected:
        explicit BasicProverCore() noexcept = default;
        ~BasicProverCore() = default;

        const RBT* rbt{nullptr};
        const RCT* rct{nullptr};
        const DBA* dba{nullptr};
        const OnCritical* on_critical{nullptr};
        const ProverSettings* settings{nullptr};
    };
}
