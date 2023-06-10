#include "convolutions.h"


__device__ __constant__ float Conv_ns::G_MASK[MASK_SIZE];
__device__ float Conv_ns::G_MASK_G[MASK_SIZE];

/* Ez a megoldás úgy konvolvál, hogy minden i-edik értéket egy-egy thread számol
 * Bemenetek:
 * 			- f: bemeneti függvény
 * 			- fn: f hossza
 * 			- y: kimeneti függvény
 * */
__global__ void Conv_ns::conv1D1(const float* f, const int fn, float* y){
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	float sum = 0.0f;

	if(id < fn){
		for(int i = 0; i <= id; i++){
			sum += f[id - i] * Conv_ns::G_MASK[i];
		}
		y[id] = sum; //nem kell semmi szinkronizáció, mert 1 thread 1 indexhez rendelhető
	}
}
__global__ void Conv_ns::conv1D1FMA(const float* f, const int fn, float* y){
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	float sum = 0.0f;

	if(id < fn){
		for(int i = 0; i <= id; i++){
			sum = __fmaf_rn(f[id - i], Conv_ns::G_MASK[i], sum);
			//sum += f[id - i] * Conv_ns::G_MASK[i];
		}
		y[id] = sum; //nem kell semmi szinkronizáció, mert 1 thread 1 indexhez rendelhető
	}
}
__global__ void Conv_ns::conv1D1G(const float* f, const int fn, float* y){
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	float sum = 0.0f;

	if(id < fn){
		for(int i = 0; i <= id; i++){
			sum += f[id - i] * Conv_ns::G_MASK_G[i];
		}
		y[id] = sum; //nem kell semmi szinkronizáció, mert 1 thread 1 indexhez rendelhető
	}
}

__global__ void conv1D1F(const float* f, const int fn, float* y){
	int id = threadIdx.x + blockIdx.x*blockDim.x;

 }

__global__ void Conv_ns::conv1D2(const float* f, const int fn, float* y){
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	float l = id < fn ? f[id] : 0.0;

	if(id < fn) y[id] = 0.0;

	if(id < fn){
		for(int i = id; i < fn; i++){
			y[i] += l * Conv_ns::G_MASK[i];
		}
	}

}
void Conv_ns::convHost(const float* f, const int fn, const float *g ,float *y){
	for(int k = 0; k < fn; k++){
		y[k] = 0;
		for (int i = 0; i < fn; i++){
			y[k] += f[k-i] * g[i];
		}
	}
}

