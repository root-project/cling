//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if a CUDA kernel works with a arguments and built-in
// functions.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

// Test, if a simple kernel with arguments works.
.rawInput 1
__global__ void gKernel1(int * out){
  *out = 42;
}
.rawInput 0

int * deviceOutput;
int hostOutput = 0;
cudaMalloc( (void **) &deviceOutput, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel1<<<1,1>>>(deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
hostOutput
// CHECK: (int) 42


// Test, if a parallel kernel with built-in functions.
const unsigned int numberOfThreads = 4;

.rawInput 1
__device__ int mul7(int  in){
  return 7*in;
}

__global__ void gKernel2(int * out){
  int i = threadIdx.x;
  out[i] = mul7(i);
}
.rawInput 0

int * deviceOutput2;
int hostOutput2[numberOfThreads];
for(unsigned int i = 0; i < numberOfThreads; ++i){
 hostOutput2[i] = 0;
}
cudaMalloc( (void **) &deviceOutput2, sizeof(int)*numberOfThreads)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel2<<<1,numberOfThreads>>>(deviceOutput2);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(hostOutput2, deviceOutput2, sizeof(int)*numberOfThreads, cudaMemcpyDeviceToHost)
// CHECK:  (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

unsigned int expectedSum = 0;
unsigned int cudaSum = 0;

for(unsigned int i = 0; i < numberOfThreads; ++i){
	expectedSum += 7*i;
	cudaSum += hostOutput2[i];
}

expectedSum == cudaSum // expected-note {{use '=' to turn this equality comparison into an assignment}}
// CHECK: (bool) true

.q
