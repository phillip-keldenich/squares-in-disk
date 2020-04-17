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

#include "ivarp/run_prover.hpp"
#include <optional>

namespace ivarp {
    extern IVARP_EXPORTED_SYMBOL bool stdout_is_tty();
    extern IVARP_EXPORTED_SYMBOL bool stderr_is_tty();
    extern IVARP_EXPORTED_SYMBOL bool stdstream_is_tty(std::ostream& stream);

    class ProgressPrinter {
    public:
        template<typename ConstraintSystemType>
        explicit ProgressPrinter(std::ostream& output, std::size_t max_num_vars,
                                 const ConstraintSystemType* arg_names) :
            output(&output), output_is_tty(stdstream_is_tty(output)),
            max_num_vars(ivarp::min(max_num_vars, std::size_t(ConstraintSystemType::num_args))), arg_names(arg_names)
        {}

        template<typename Context, typename EntryType>
        void observe_progress(const Context& /*ctx*/, const EntryType& queue_head,
                              const ProgressInfo& info)
        {
            std::ostringstream buffer;
            buffer << "Queue size: " << info.queue_size << ", cuboids: " << info.cuboid_count << "("
                   << info.critical_count << ')';
            for(std::size_t i = 0; i < max_num_vars; ++i) {
                buffer << ", " << arg_names->arg_name(i) << ": " << queue_head.elements[i];
            }
            std::string str = buffer.str();
            if(str.size() > max_chars_printed) {
                max_chars_printed = str.size();
            } else {
                if(output_is_tty) {
                    str.append(max_chars_printed - str.size(), ' ');
                }
            }
            if(output_is_tty) {
                str.append(max_chars_printed, '\b');
            } else {
                str += '\n';
            }
            *output << str;
            if(output_is_tty) {
                output->flush();
            }
        }

        void observe_done() const noexcept {
            if(max_chars_printed > 0 && output_is_tty) {
                std::string buf(max_chars_printed, ' ');
                buf.append(max_chars_printed, '\b');
                *output << buf << std::flush;
            }
        }

        void observe_error(std::exception_ptr error, std::size_t error_thread_id) {
            if(max_chars_printed > 0 && output_is_tty) {
                std::string buf(max_chars_printed, ' ');
                buf.append(max_chars_printed, '\b');
                *output << buf << std::flush;
            }
            *output << "Proof aborting with error thrown by prover thread #" << error_thread_id << std::endl;
        }

    private:
        std::ostream* output;
        bool output_is_tty;
        std::size_t max_num_vars;
        std::size_t max_chars_printed{0};
        const ArgNameLookup* arg_names;
    };
}
