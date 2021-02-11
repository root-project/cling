//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks whether a define argument (-DTEST=3) is passed to the PTX
// compiler. If it works, it should not throw an error.
// RUN: cat %s | %cling -DTEST=3 -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include <iostream>

// Check if cuda driver is available
int version;
cudaDriverGetVersion(&version)
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0

// Check if a CUDA compatible device (GPU) is available.
int device_count = 0;
cudaGetDeviceCount(&device_count)
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
device_count > 0
// CHECK: (bool) true

TEST
// CHECK: (int) 3

.rawInput 1

__global__ void g(){
    int i = TEST;
}

.rawInput 0

// Runing the kernel is neccessary because FileCheck has problems whith the
// error output of the PTX compiler. Therefore I need an error message from
// the host interpreter.
g<<<1,1>>>();
cudaGetLastError()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0

// expected-no-diagnostics
.q
