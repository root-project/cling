//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if cuda streams works.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

const unsigned int numberOfThreads = 4;

.rawInput 1
__global__ void gKernel1(int * a, int b){
  int i = threadIdx.x;
  a[i] += b;
}
.rawInput 0

cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaStreamCreate(&stream2)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

int hostInput1[numberOfThreads];
int hostInput2[numberOfThreads];
int hostOutput1[numberOfThreads];
int hostOutput2[numberOfThreads];
for(unsigned int i = 0; i < numberOfThreads; ++i){
	hostInput1[i] = i;
	hostInput2[i] = i+10;
}
int * device1;
int * device2;
cudaMalloc( (void **) &device1, sizeof(int)*numberOfThreads)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMalloc( (void **) &device2, sizeof(int)*numberOfThreads)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpyAsync(device1, hostInput1, sizeof(int)*numberOfThreads, cudaMemcpyHostToDevice, stream1)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpyAsync(device2, hostInput2, sizeof(int)*numberOfThreads, cudaMemcpyHostToDevice, stream2)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel1<<<1,numberOfThreads,0,stream2>>>(device2, 2);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel1<<<1,numberOfThreads,0,stream1>>>(device1, 1);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpyAsync(hostOutput2, device2, sizeof(int)*numberOfThreads, cudaMemcpyDeviceToHost, stream2)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpyAsync(hostOutput1, device1, sizeof(int)*numberOfThreads, cudaMemcpyDeviceToHost, stream1)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

unsigned int expectedSum1 = 0;
unsigned int cudaSum1 = 0;
unsigned int expectedSum2 = 0;
unsigned int cudaSum2 = 0;

for(unsigned int i = 0; i < numberOfThreads; ++i){
	expectedSum1 += i+1;
	cudaSum1 += hostOutput1[i];
	expectedSum2 += i+12;
	cudaSum2 += hostOutput2[i];
}

expectedSum1 == cudaSum1 // expected-note {{use '=' to turn this equality comparison into an assignment}}
// CHECK: (bool) true
expectedSum2 == cudaSum2 // expected-note {{use '=' to turn this equality comparison into an assignment}}
// CHECK: (bool) true

.q
