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

add_executable(squares_in_disk_prover
	main.cpp top_packing.cpp one_subcontainer.cpp two_subcontainer.cpp three_subcontainer.cpp
	four_subcontainer.cpp five_subcontainer.cpp six_subcontainer.cpp seven_subcontainer.cpp)
target_link_libraries(squares_in_disk_prover PRIVATE ivarp)

if(CMAKE_CUDA_COMPILER AND NOT IVARP_NO_CUDA)
	add_executable(squares_in_disk_prover_cuda EXCLUDE_FROM_ALL
		main.cpp top_packing.cu one_subcontainer.cu two_subcontainer.cu 
		three_subcontainer.cu four_subcontainer.cu five_subcontainer.cu
		six_subcontainer.cu seven_subcontainer.cu)
	target_link_libraries(squares_in_disk_prover_cuda PRIVATE ivarp)
endif()

