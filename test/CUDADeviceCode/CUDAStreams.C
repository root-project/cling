//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if cuda streams works.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

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

int host1[] = {1,2,3,4};
int host2[] = {11,12,13,14};
int * device1;
int * device2;
cudaMalloc( (void **) &device1, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMalloc( (void **) &device2, sizeof(int)*4)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpyAsync(device1, &host1, sizeof(int)*4, cudaMemcpyHostToDevice, stream1)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpyAsync(device2, &host2, sizeof(int)*4, cudaMemcpyHostToDevice, stream2)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel1<<<1,4,0,stream2>>>(device2, 2);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel1<<<1,4,0,stream1>>>(device1, 1);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpyAsync(&host2, device2, sizeof(int)*4, cudaMemcpyDeviceToHost, stream2)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpyAsync(&host1, device1, sizeof(int)*4, cudaMemcpyDeviceToHost, stream1)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
host1[0] + host1[1] + host1[2] + host1[3]
// CHECK: (int) 14
host2[0] + host2[1] + host2[2] + host2[3]
// CHECK: (int) 58


// expected-no-diagnostics
.q
