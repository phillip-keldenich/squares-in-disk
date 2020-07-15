# Packing Squares into a Disk with Optimal Worst-Case Density: Automatic Proofs
This repository contains the code for the automatic proofs of our paper 

> Packing Squares into a Disk with Optimal Worst-Case Density

submitted to SoCG 2020.

## Getting and Building the Code
The code makes use of our interval arithmetic prover framework at 

> https://gitlab.ibr.cs.tu-bs.de/alg/ivarp.git

that is included in this repository as a git submodule.
Therefore, when cloning, use 

    git clone --recursive https://gitlab.ibr.cs.tu-bs.de/alg/square-in-circle-proofs.git

to automatically clone this dependency as well.

## Further Dependencies
The code depends on the GNU Multiple Precision Arithmetic Library (GMP, see https://gmplib.org/) and the 
GNU Multiple Precision Floating-Point Reliable Library (MPFR, see https://www.mpfr.org/).

These LGPL-licensed open source libraries are not included in this repository and are not developed by us.
On Unix-like systems (Linux, MacOS), these libraries can be obtained by package
managers (like apt for Ubuntu or HomeBrew for MacOS).
On Windows, building these libraries from source code is not officially supported;
there are, however, forks of these libraries (under the name MPIR, see http://mpir.org/), for which
Visual Studio projects are available.

Furthermore, the code also depends on (the header-only part of) the open source Boost C++ libraries (see https://www.boost.org/).
Like the other libraries, these are not included in this repository and are not developed by us.

## Building the Code
The code is built using the CMake Cross-Platform Make toolkit available at https://cmake.org/ .

On a Unix-like system, building typically consists of the following sequence of commands (within the root directory of the repository).

    mkdir cmake-build-release && cd cmake-build-release
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

Compilation may take some time.
These commands should generate an executable running the proof at `${PROJECT_ROOT}/cmake-build-release/src/squares_in_disk_prover`.

## Running the Code
The executable `squares_in_disk_prover` can be run without an argument.

### Example Output
The output of the program contains informs about the progress of the prover and the number of individual hypercuboids considered.

    --> time ./squares_in_disk_prover 
    Starting proof Lemma 30, statement (1)...
    Done: 256 cuboids considered.
    Starting proof Lemma 30, statement (2)...
    Done: 309 cuboids considered.
    Starting proof Lemma 31, statement (1)...
    Done: 1933541 cuboids considered.
    Starting proof Lemma 31, statement (2)...
    Done: 12829 cuboids considered.
    Starting proof Lemma 31, statement (3)...
    Done: 72974 cuboids considered.
    Starting proof Lemma 31, statement (4)...
    Done: 32736 cuboids considered.
    Starting proof Lemma 31, statement (5)...
    Done: 5590 cuboids considered.
    Starting proof Lemma 32, statement (1)...
    Done: 116367 cuboids considered.
    Starting proof Lemma 32, statement (2)...
    Done: 116523 cuboids considered.
    Starting proof Lemma 33, statement (1)...
    Done: 1713675 cuboids considered.
    Starting proof Lemma 33, statement (2)...
    Done: 678625 cuboids considered.
    Starting proof Lemma 34, statement (1)...
    Done: 16781312 cuboids considered.
    Starting proof Lemma 34, statement (2)...
    Done: 635262259 cuboids considered.
    Starting proof Lemma 34, statement (3)...
    Done: 8460712 cuboids considered.
    Starting proof Lemma 35, N = 5...
    Done: 1104 cuboids considered.
    Starting proof Lemma 35, N = 6...
    Done: 1024 cuboids considered.
    Starting proof Lemma 35, N = 7...
    Done: 1024 cuboids considered.
    ./squares_in_disk_prover  10877,83s user 7,04s system 778% cpu 23:18,96 total

## License
The code is open source under MIT license.
Copyright TU Braunschweig, Algorithms Group, https://ibr.cs.tu-bs.de/alg

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.