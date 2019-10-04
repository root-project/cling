//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The test checks whether setting a new include path at runtime also works for
// the PTX compiler.
// RUN: cat %s | %cling -DTEST_PATH="\"%/p/\"" -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include <iostream>

// Check if cuda driver is available
int version;
cudaDriverGetVersion(&version)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// Check if a CUDA compatible device (GPU) is available.
int device_count = 0;
cudaGetDeviceCount(&device_count)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
device_count > 0
// CHECK: (bool) true

#include "cling/Interpreter/Interpreter.h"
gCling->AddIncludePaths(TEST_PATH "include");

#include "foo.h"

foo()
// CHECK: (int) 3

// Runing the kernel is neccessary because FileCheck has problems whith the
// error output of the PTX compiler. Therefore I need an error message from
// the host interpreter.
bar<<<1,1>>>();
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0


// expected-no-diagnostics
.q
