#include "matrix.h"
#include <iostream>



void VecAdd_h(float* A, float* B, float* C, int N) {
	for (int i = 0; i < N; i++) {
		C[i] = A[i] + B[i];
	}
}

__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}

__global__ void ArraySum(float* v, float* result, int N) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ float psum;

	if (threadIdx.x == 0) {
		psum = 0.0f;
	}
	__syncthreads();
	if (id < N) {
		atomicAdd(&psum, v[id]);
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(result,psum);
	}
}
__global__ void ArraySumBin(float* v, float* result, int N) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ float  sarray[256];
	sarray[threadIdx.x] = id < N ? v[id] : 0.0f;  //bet�lt�m a shared mem�ri�ba thread-enk�nt
	
	for (int i = blockDim.x / 2;i > 0;i /= 2) {
		__syncthreads();    //minding szinkronizlni kell, mert egym�s eredm�nyeit�l f�ggnek a sz�m�t�sok (az 1-1 p�r eredm�ny�t�l)
		if (threadIdx.x < i) sarray[threadIdx.x] += sarray[threadIdx.x + i];
	}
	if (threadIdx.x == 0x00) atomicAdd(result,sarray[0]);  //a v�g�n blokkonk�nt hozz�adom az eredm�nyt
}
void ArraySum_h(float* v, float* result, int N) {
	*result = 0;
	for (int i = 0; i < N;i++) {
		*result += v[i];
	}
}

void MatAdd_h(Matrix A, Matrix B, Matrix C) {
	for (int i = 0; i < A.height; i++) {
		for (int j = 0; j < A.width; j++) {
			C.elements[i * A.width + j] = A.elements[i * A.width + j] + B.elements[i * A.width + j];
		}
	}
}

__global__ void MatAdd(Matrix A, Matrix B, Matrix C) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	
	if (row < A.height && col < A.width) {
		C.elements[row * A.width + col] = A.elements[row * A.width + col] + B.elements[row * A.width + col];
	}
}
__global__ void MatAddFMA(Matrix A, Matrix B, Matrix C) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < A.height && col < A.width) {
		C.elements[row * A.width + col] = A.elements[row * A.width + col] + B.elements[row * A.width + col];
	}
}

__global__ void MatMulSimple(Matrix A, Matrix B, Matrix C) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < C.height && col < C.width) {
		float sum = 0.0f;
		for (int i = 0; i < A.width; i++) {
			sum += A.elements[row * A.width + i] * B.elements[i * B.width + col];
		}
		C.elements[row * C.width + col] = sum;
	}
}
__global__ void MatMulSimpleFMA(Matrix A, Matrix B, Matrix C) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < C.height && col < C.width) {
		float sum = 0.0f;
		for (int i = 0; i < A.width; i++) {
			sum = __fmaf_rn(A.elements[row * A.width + i], B.elements[i * B.width + col], sum);
		}
		C.elements[row * C.width + col] = sum;
	}
}
void MatMulHost(Matrix A, Matrix B, Matrix C) {
	for (int i = 0; i < C.height; i++) {
		for (int j = 0;j < C.width;j++) {
			for (int k = 0; k < A.width;k++) {
				C.elements[i * C.width + j] += A.elements[i * A.width + k] * B.elements[k * B.width + j];
			}
		}
	}
}



void printVec(float* v, int N) {
	for (int i = 0; i < N; i++) {
		std::cout << "\t" << v[i] << std::endl;
	}
}

void initVec(float* v, float val, int N) {
	for (int i = 0; i < N;i++) {
		v[i] = val;
	}
}

void printMat(Matrix A, int N1, int N2) {
	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			std::cout << A.elements[i * A.width + j] << "\t";
		}
		std::cout << std::endl;
	}
}
void initMat(Matrix* A, float val) {
	A->elements = (float*) malloc((sizeof(float)) * (A->width) * (A->height));
	for (int i = 0; i < A->height; i++) {
		for (int j = 0; j < A->width; j++) {
			A->elements[i * A->width + j] = val;
		}
	}
}

void initMat_d(Matrix* A) {
	cudaMalloc(&(A->elements),(sizeof(float)) * (A->width) * (A->height));
}
