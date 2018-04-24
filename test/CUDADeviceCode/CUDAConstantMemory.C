//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if constant memory works.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

__constant__ int constNum[4];

.rawInput 1
__global__ void gKernel1(int * output){
  int i = threadIdx.x;
  output[i] = constNum[i];
}

int hostInput[4] = {1,2,3,4};
int hostOutput[4] = {0,0,0,0};
int * deviceOutput;
cudaMalloc( (void **) &deviceOutput, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpyToSymbol(constNum, &hostInput, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel1<<<1,4>>>(deviceOutput);
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
