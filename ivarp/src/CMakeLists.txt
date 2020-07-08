# The code is open source under the MIT license.
# Copyright 2019-2020, Phillip Keldenich, TU Braunschweig, Algorithms Group
# https://ibr.cs.tu-bs.de/alg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_library(__ivarp_sources INTERFACE)

set(IVARP_LIB_SOURCES_NAMES
		init_rounding.cpp convert_number.cpp
		powint.cpp irr_constants.cpp
		rational_invert.cpp rational_modf.cpp
		irational_mul.cpp irational_div.cpp
		interval_asin.cpp interval_acos.cpp
		mpfr_helpers.hpp
		interval_sin.cpp sincos_period_info.hpp
		interval_cos.cpp irational_sqrt.cpp
		fp_print.cpp progress_printer.cpp
		factorize.cpp)

set(IVARP_LIB_SOURCES "")
foreach(a IN LISTS IVARP_LIB_SOURCES_NAMES)
	list(APPEND IVARP_LIB_SOURCES "${CMAKE_CURRENT_LIST_DIR}/${a}")
endforeach()

target_sources(__ivarp_sources INTERFACE ${IVARP_LIB_SOURCES})
source_group(ivarp\\shared_lib FILES ${IVARP_LIB_SOURCES})
