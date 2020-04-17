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
// Created by Phillip Keldenich on 17.02.20.
//

#include "ivarp/number.hpp"
#include <iomanip>
#include <sstream>
#include <string>
#include <ivarp/prover.hpp>

std::ostream& ivarp::fixed_point_bounds::operator<<(std::ostream& output, PrintFixedPoint p) {
    if(p.value >= max_bound()) {
        output << "inf";
    } else if(p.value <= min_bound()) {
        output << "-inf";
    } else {
        if(p.value < 0) {
            output << '-';
            p.value = -p.value;
        }

        std::int64_t intpart = p.value / denom();
        std::int64_t fracpart = p.value % denom();

        output << intpart;
        if(fracpart != 0) {
            std::stringstream s;
            s << '.' << std::setfill('0') << std::setw(6) << fracpart;
            std::string str = s.str();
            while(str.back() == '0') {
                str.pop_back();
            }
            output << str;
        }
    }
    return output;
}

auto ivarp::FunctionPrinter::get_default_printer() -> FunctionPrinter& {
    static DefaultArgNameLookup names;
    static FunctionPrinter printer(PrintOptions{}, &names);
    return printer;
}
