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

#include <cstdio>
#include <iostream>

namespace {
    /* initialize cout/cerr/clog buffers */
    static std::streambuf const * const coutbuf = std::cout.rdbuf();
    static std::streambuf const * const cerrbuf = std::cerr.rdbuf();
    static std::streambuf const * const clogbuf = std::clog.rdbuf();
}

#ifdef _MSC_VER
#include <io.h>

namespace {
    int get_stdout_fd() {
        using namespace std;
        return _fileno(stdout);
    }

    int get_stderr_fd() {
        using namespace std;
        return _fileno(stderr);
    }

    bool fd_names_tty(int fd) {
        return _isatty(fd) != 0;
    }
}

#else
#include <unistd.h>

namespace {
    constexpr int get_stdout_fd() {
        return STDOUT_FILENO;
    }

    constexpr int get_stderr_fd() {
        return STDERR_FILENO;
    }

    bool fd_names_tty(int fd) {
        return isatty(fd) != 0;
    }
}

#endif

#include "ivarp/progress_printer.hpp"

bool ivarp::stdout_is_tty() {
    return fd_names_tty(get_stdout_fd());
}

bool ivarp::stderr_is_tty() {
    return fd_names_tty(get_stderr_fd());
}

bool ivarp::stdstream_is_tty(std::ostream& output) {
    if(output.rdbuf() == coutbuf) {
        return stdout_is_tty();
    }
    if(output.rdbuf() == cerrbuf || output.rdbuf() == clogbuf) {
        return stderr_is_tty();
    }
    return false;
}
