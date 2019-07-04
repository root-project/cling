//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The test checks if the interface functions process(), declare() and parse()
// of cling::Interpreter also work in the cuda mode.
// RUN: cat %s | %cling -DTEST_PATH="\"%/p/\"" -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include "cling/Interpreter/Interpreter.h"

// if process() works, the general input also works
gCling->process("cudaGetLastError()");
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// declare a cuda kernel with with a define
// do not this in real code ;-)
gCling->declare("#define FOO 42");
gCling->declare("__global__ void g1(int * out){ *out = FOO;}");

// allocate memory on CPU and GPU
int *d1;
int h1 = 0;
cudaMalloc((void**)&d1, sizeof(int))
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// run kernel
g1<<<1,1>>>(d1);
cudaGetLastError()
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// check result
cudaMemcpy(&h1, d1, sizeof(int), cudaMemcpyDeviceToHost)
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
h1
//CHECK: (int) 42

// same test as declare()
// reuse memory
// FIXME: at the moment there is no check whether the device compiler generates
// code or not
// to fix the problem, cling needs a debug interface for the device compiler
gCling->parse("__global__ void g2(int * out){ *out = 52;}");

g2<<<1,1>>>(d1);
cudaGetLastError()
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(&h1, d1, sizeof(int), cudaMemcpyDeviceToHost)
//CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
h1
//CHECK: (int) 52

cudaFree(d1)

// expected-no-diagnostics
.q
