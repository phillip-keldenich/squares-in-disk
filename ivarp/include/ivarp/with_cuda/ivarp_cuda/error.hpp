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
// Created by Phillip Keldenich on 25.11.19.
//

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <exception>
#include <string>
#include <sstream>
#include <iostream>

namespace ivarp {
    class CUDAError : public std::exception {
    public:
        explicit CUDAError(const std::string& msg, cudaError_t errcode) : error_code(errcode) {
            std::ostringstream out;
            out << msg << ": " << cudaGetErrorName(errcode) << " (code " << static_cast<int>(error_code) << ')'
                << " - " << cudaGetErrorString(errcode);
            message = out.str();
        }

        const char* what() const noexcept override {
            return message.c_str();
        }

        cudaError_t get_error_code() const noexcept {
            return error_code;
        }

    private:
        std::string    message;
        cudaError_t error_code;
    };

    static inline void throw_if_cuda_error(const std::string& msg, cudaError_t error_code) {
        if(error_code != cudaSuccess) {
            cudaGetLastError();
            throw CUDAError(msg, error_code);
        }
    }

    static inline void warn_if_cuda_error(const std::string& msg, cudaError_t error_code) {
        if(error_code != cudaSuccess) {
            cudaGetLastError();
            std::cerr << msg << ": " << cudaGetErrorName(error_code) << " (code " << static_cast<int>(error_code) << ')'
                      << " - " << cudaGetErrorString(error_code);
        }
    }
}
