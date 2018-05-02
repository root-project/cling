//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if runtime shared memory works.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

const unsigned int numberOfThreads = 4;

.rawInput 1
__global__ void gKernel1(int * input, int * output){
  extern __shared__ int s[];
  int i = threadIdx.x;
  s[(i+1)%blockDim.x] = input[i];
  __syncthreads();
  output[i] = s[i];
}
.rawInput 0

int hostInput[numberOfThreads];
int hostOutput[numberOfThreads];
for(unsigned int i = 0; i < numberOfThreads; ++i){
	hostInput[i] = i+1;
	hostOutput[i] = 0;
}
int * deviceInput;
int * deviceOutput;
cudaMalloc( (void **) &deviceInput, sizeof(int)*numberOfThreads)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMalloc( (void **) &deviceOutput, sizeof(int)*numberOfThreads)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(deviceInput, hostInput, sizeof(int)*numberOfThreads, cudaMemcpyHostToDevice)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel1<<<1,numberOfThreads, sizeof(int)*numberOfThreads>>>(deviceInput, deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(hostOutput, deviceOutput, sizeof(int)*numberOfThreads, cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

int expectedSum = (numberOfThreads*(numberOfThreads+1))/2;
int cudaSum = 0;
 
for(unsigned int i = 0; i < numberOfThreads; ++i){
	cudaSum += hostOutput[i];
}

//check, if elements was shifted
// small workaround, to avoid compiler hint '='
bool result1 = hostOutput[0] == numberOfThreads
// CHECK: (bool) true
bool result2 = hostOutput[numberOfThreads-1] == numberOfThreads-1
// CHECK: (bool) true
bool result3 = expectedSum == cudaSum
// CHECK: (bool) true



// expected-no-diagnostics
.q