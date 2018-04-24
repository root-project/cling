//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if global __device__ memory works. There two tests. One use
// direct value assignment at declaration and the other use a reassignment.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

__device__ int dAnswer = 42;

.rawInput 1
__global__ void gKernel1(int * output){
  int i = threadIdx.x;
  output[i] = dAnswer;
}
.rawInput 0

int hostOutput[4] = {1,1,1,1};
int * deviceOutput;
cudaMalloc( (void **) &deviceOutput, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel1<<<1,4>>>(deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int)*4, cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// FIXME: output of the whole static array isn't working at the moment
hostOutput[0]
// CHECK: (int) 42
hostOutput[1]
// CHECK: (int) 42
hostOutput[2]
// CHECK: (int) 42
hostOutput[3]
// CHECK: (int) 42

// Test, if value assignment also works.
dAnswer = 43;

gKernel1<<<1,4>>>(deviceOutput);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput, deviceOutput, sizeof(int)*4, cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

// FIXME: output of the whole static array isn't working at the moment
hostOutput[0]
// CHECK: (int) 43
hostOutput[1]
// CHECK: (int) 43
hostOutput[2]
// CHECK: (int) 43
hostOutput[3]
// CHECK: (int) 43


// expected-no-diagnostics
.q
