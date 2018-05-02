//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks if templated CUDA kernel in some special cases works.
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

// Check if templated CUDA kernel works, without explicit template type declaration.
.rawInput 1

template <typename T>
__global__ void gKernel1(T * value){
  *value = (T)42.0;
}
.rawInput 0

int * deviceOutput1;
int hostOutput1 = 1;
cudaMalloc( (void **) &deviceOutput1, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel1<<<1,1>>>(deviceOutput1);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput1, deviceOutput1, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
hostOutput1
// CHECK: (int) 42



// Check if specialization of templated CUDA kernel works.

.rawInput 1

template <typename T>
__global__ void gKernel2(T * value){
  *value = (T)1.0;
}

template <>
__global__ void gKernel2<float>(float * value){
  *value = 2.0;
}
.rawInput 0

int * deviceOutput2;
int hostOutput2 = 10;
cudaMalloc( (void **) &deviceOutput2, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
float * deviceOutput3;
float hostOutput3= 10.0;
cudaMalloc( (void **) &deviceOutput3, sizeof(float))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel2<<<1,1>>>(deviceOutput2);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel2<<<1,1>>>(deviceOutput3);
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(&hostOutput2, deviceOutput2, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput3, deviceOutput3, sizeof(float), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

hostOutput2
// CHECK: (int) 1
hostOutput3
// CHECK: (float) 2.00000f



// Check if function as parameter works on a templated CUDA kernel.
.rawInput 1

template <typename T>
__global__ void gKernel3(T * out, int value){
  *out = value;
}

__global__ void gKernel4(int * out){
  *out = 5;
}

int func1(int * input){
	int result = 1;
	gKernel4<<<1,1>>>(input);
	cudaMemcpy(&result, input, sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}
.rawInput 0

int * deviceOutput4;
int hostOutput4 = 10;
cudaMalloc( (void **) &deviceOutput4, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
int * deviceOutput5;
cudaMalloc( (void **) &deviceOutput5, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel3<<<1,1>>>(deviceOutput4, func1(deviceOutput5));
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(&hostOutput4, deviceOutput4, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

hostOutput4
// CHECK: (int) 5

// Check if specialization of struct __device__ functors works.

template<typename T>
struct Struct1
{
	__device__ T operator()(T* dummy) const
	{
		return (T)1;
	}
};

template<>
struct Struct1<double>
{
	__device__ double operator()(double * dummy) const
	{
		return 2.0;
	}
};

.rawInput 1

template<typename T, typename Functor>
__global__ void gKernel5(T * x, Functor const functor){
	*x = functor(x);
}
.rawInput 0

int * deviceOutput6;
int hostOutput6 = 10;
cudaMalloc( (void **) &deviceOutput6, sizeof(int))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
double * deviceOutput7;
double hostOutput7 = 10.0;
cudaMalloc( (void **) &deviceOutput7, sizeof(double))
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

gKernel5<<<1,1>>>(deviceOutput6, Struct1<int>{});
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
gKernel5<<<1,1>>>(deviceOutput7, Struct1<double>{});
cudaGetLastError()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaDeviceSynchronize()
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

cudaMemcpy(&hostOutput6, deviceOutput6, sizeof(int), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0
cudaMemcpy(&hostOutput7, deviceOutput7, sizeof(double), cudaMemcpyDeviceToHost)
// CHECK: (cudaError_t) (cudaError::cudaSuccess) : (unsigned int) 0

hostOutput6
// CHECK: (int) 1

hostOutput7
// CHECK: (double) 2.0000000

// expected-no-diagnostics
.q