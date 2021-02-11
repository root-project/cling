//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if templated CUDA kernel works.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

// Check if template device side resoultion works.
.rawInput 1
template <int T>
__device__ int dKernel1(){
  return T;
}

__global__ void gKernel1(int * out){
  *out = dKernel1<42>();
}
.rawInput 0

int * deviceOutput;
int hostOutput = 0;
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


// Check if template host-device side resoultion works.
.rawInput 1
template <int T>
__global__ void gKernel2(int * out){
  *out = dKernel1<T>();
}
.rawInput 0

hostOutput = 0;
gKernel2<43><<<1,1>>>(deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaSuccess) : (unsigned int) 0
hostOutput
// CHECK: (int) 43


// expected-no-diagnostics
.q
