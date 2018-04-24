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

.rawInput 1
__global__ void gKernel1(int * input, int * output){
  extern __shared__ int s[];
  int i = threadIdx.x;
  s[i] = input[i];
  output[i] = s[i];
}
.rawInput 0

int hostInput[4] = {1,2,3,4};
int hostOutput[4] = {0,0,0,0};
int * deviceInput;
int * deviceOutput;
cudaMalloc( (void **) &deviceInput, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMalloc( (void **) &deviceOutput, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(deviceInput, &hostInput, sizeof(int)*4, cudaMemcpyHostToDevice)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel1<<<1,4, 4 * sizeof(int)>>>(deviceInput, deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int)*4, cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
 
// FIXME: output of the whole static array isn't working at the moment
hostOutput[0]
// CHECK: (int) 1
hostOutput[1]
// CHECK: (int) 2
hostOutput[2]
// CHECK: (int) 3
hostOutput[3]
// CHECK: (int) 4


// expected-no-diagnostics
.q