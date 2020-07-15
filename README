# Packing Squares into a Disk with Optimal Worst-Case Density: Automatic Proofs
This repository contains the code for the automatic proofs of our paper 

> Packing Squares into a Disk with Optimal Worst-Case Density

submitted to SODA 2021.

## Getting and Building the Code
The code makes use of our interval arithmetic prover framework at 

> https://gitlab.ibr.cs.tu-bs.de/alg/ivarp.git

that is included as a copy in this repository.

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

Compilation may take quite some time.
These commands should generate an executable running the proof at `${PROJECT_ROOT}/cmake-build-release/src/squares_in_disk_prover`.

## Running the Code
The executable `squares_in_disk_prover` can be run without command line arguments.

### Example Output
The output of the program contains informs about the progress of the prover and the number of individual hypercuboids considered.

    `--> time ./src/squares_in_disk_prover
    Starting proof Top Packing Lemma, statement (1)...
    Done: 128 cuboids considered (128 leafs).
    Starting proof Top Packing Lemma, statement (2)...
    Done: 168 cuboids considered (168 leafs).
    Starting proof One Subcontainer Lemma...
    Done: 2869888 cuboids considered (2858678 leafs).
    Starting proof Two Subcontainer Lemma...
    Done: 7181696 cuboids considered (7069587 leafs).
    Starting proof Three Subcontainer Lemma...
    Done: 3549856 cuboids considered (3440786 leafs).
    Starting proof Four Subcontainer Lemma...
    Done: 37245328 cuboids considered (35026303 leafs).                                                                                                                                      
    Starting proof Five Subcontainers, s_n > sigma...
    Done: 3392 cuboids considered (3188 leafs).
    Starting proof Six Subcontainers, s_n > sigma...
    Done: 14848 cuboids considered (13928 leafs).
    Starting proof Seven Subcontainers, s_n > sigma...
    Done: 1808 cuboids considered (1703 leafs).
    Starting proof >= 5 Subcontainers, s_n <= sigma, y_3 <= 0...
    Done: 1056154896 cuboids considered (1022752013 leafs).                                                                                                                                      
    Starting proof >= 5 Subcontainers, s_n <= sigma, y_3 > 0...
    Done: 123127968 cuboids considered (115680431 leafs).                                                                                                                                    
    ./src/squares_in_disk_prover  27306,87s user 36,52s system 757% cpu 1:00:08,12 total

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
