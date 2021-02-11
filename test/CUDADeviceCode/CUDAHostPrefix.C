//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if a function with __host__ and __device__ prefix available
// on host and device side.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

.rawInput 1
__host__ __device__ int sum(int a, int b){
  return a + b;
}

__global__ void gKernel1(int * output){
  *output = sum(40,2);
}
.rawInput 0

sum(41,1)
// CHECK: (int) 42

int hostOutput = 0;
int * deviceOutput;
cudaMalloc( (void **) &deviceOutput, sizeof(int))
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0

gKernel1<<<1,1>>>(deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0

hostOutput
// CHECK: (int) 42


// expected-no-diagnostics
.q
