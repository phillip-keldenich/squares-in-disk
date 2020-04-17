#!/bin/sh
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

if [ "$#" -eq 2 ]
then

SOURCE_DIR=$1
TARGET_DIR=$2
if [ ! -d ${TARGET_DIR} ]
then
mkdir -p ${TARGET_DIR}
fi

if [ -d ${SOURCE_DIR} ]
then
lcov -c -d ${SOURCE_DIR} -o "${TARGET_DIR}/cov.all.info" && lcov --remove "${TARGET_DIR}/cov.all.info" -o "${TARGET_DIR}/cov.info" '*test_util.hpp' '/usr/include/*' '/usr/local/include/*' '/Library/*' '*doctest*' && \
	genhtml "${TARGET_DIR}/cov.info" -o "${TARGET_DIR}" 
else
echo "Error: Source directory does not exist!"
exit 1
fi

else
echo "Error: Expected two arguments (directory to work in and output directory)."
exit 1
fi
