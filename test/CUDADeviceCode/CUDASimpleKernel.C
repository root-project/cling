//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if a CUDA compatible device is available and checks, if simple
// __global__ and __device__ kernels are running.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

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

// Check, if the smallest __global__ kernel is callable.
.rawInput 1
__global__ void gKernel1(){}
.rawInput 0
gKernel1<<<1,1>>>();
cudaGetLastError()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0

// Check, if a simple __device__ kernel is useable.
.rawInput 1
__device__ int dKernel1(){return 42;}
__global__ void gKernel2(){int i = dKernel1();}
.rawInput 0
gKernel2<<<1,1>>>();
cudaGetLastError()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0


// expected-no-diagnostics
.q
