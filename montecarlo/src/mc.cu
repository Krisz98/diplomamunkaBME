/*
 * mc.cu
 *
 *  Created on: May 4, 2023
 *      Author: krisztian
 */

#include "mc.h"
#include "cmath"
#include "stdio.h"
#define VECTOR_SIZE 32


__global__ void setupmon(curandState *state, int N, int *j){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	//if(id==0)printf(" CHECKPOINT ");
	if (id < N)curand_init(12334 + *j, id, 0, &state[id]);
	if (id ==0)printf("alive: %d",*j);
}
__global__ void setupmonPhil(curandStatePhilox4_32_10_t *state, int N, int *j){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)curand_init(1234 + *j, id, 0, &state[id]);
}
__global__ void setupmonMrg(curandStateMRG32k3a *state, int N, int *j){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)curand_init(1234 + *j, id, 0, &state[id]);
}
__global__ void setupmonSobol(curandStateSobol32 *state, unsigned int *direction_vectors, int N, int *j, int ndim){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < N){
		int dim = ndim * id;
		for(int i = 0; i < ndim; i++){
			curand_init(direction_vectors + VECTOR_SIZE*i,
							id + (*j),
							&state[dim + i]);
		}
	}
}

__device__ float func(float x){
	return sin(x);
}
__device__ float funcf(float x){
	return sinf(x);
}
__device__ float fop(float *vs){
	//return sin(vs[0]);
	return vs[0]*sin(vs[0]*vs[0])*vs[1]*vs[1];
	//return vs[0]*vs[0]*vs[0] + 2.0*vs[0]*vs[1] - 4.0 * vs[2]*vs[3] + 3.0*vs[3];
}
__device__ float d_fop(double *vs){
	//return sin(vs[0]);
	return vs[0]*sin(vs[0]*vs[0])*vs[1]*vs[1];
	//return vs[0]*vs[0]*vs[0] + 2.0*vs[0]*vs[1] - 4.0 * vs[2]*vs[3] + 3.0*vs[3];
}

__global__ void midpointrule(int N, float *result, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b-a)/N;
	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();

	if(id < N){
		float xi, xip1;
		xi = a + h * id;
		xip1 = xi + h;
		sums[threadIdx.x] = h * func((xi + xip1) / 2.0f);

		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;
			atomicAdd(result, sums[0]);
		}
		__syncthreads();

	}
}
__global__ void midpointruleFMA(int N, float *result, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b-a)/N;
	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();

	if(id < N){
		float xi, xip1;
		//xi = a + h * id;
		xi = __fmaf_rn(h, id,a);
		xip1 = xi + h;
		sums[threadIdx.x] = h * funcf((xi + xip1) / 2.0f);

		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;

			atomicAdd(result, sums[0]);
		}
		__syncthreads();

	}
}
__global__ void trapezoidrule(int N, float *result, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b-a)/N;
	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();

	if(id < N){
		float xim1, xi;
		xim1 = a + h * id;
		xi = xim1 + h;
		sums[threadIdx.x] = h * (func(xi) + func(xim1)) / 2.0f;

		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;
			atomicAdd(result, sums[0]);
		}
		__syncthreads();

	}
}
__global__ void simpsonsrule(int N, float *result, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b-a)/N;
	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();

	if(id < N){
		float xi0, xi1, xi2;
		xi0 = a + h * id;
		xi1 = xi0 + 0.5f * h;
		xi2 = xi0 + h;
		sums[threadIdx.x] = h * (func(xi0) +  4 * func(xi1) + func(xi2)) / 6.0f;

		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;
			atomicAdd(result, sums[0]);
		}
		__syncthreads();

	}
}
__global__ void simpsonsruleFMA(int N, float *result, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float h = (b-a)/N;
	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();

	if(id < N){
		float xi0, xi1, xi2;
		xi0 = __fmaf_rn(h, id, a);
		xi1 = __fmaf_rn(0.5f, h, xi0);
		xi2 = xi0 + h;
		//sums[threadIdx.x] = h * (func(xi0) +  4 * func(xi1) + func(xi2)) / 6.0f;
		float s = __fmaf_rn(4.0f * h, func(xi1), h * func(xi2));
		sums[threadIdx.x] = __fmaf_rn(h, func(xi0), s) / 6.0f;
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}
		__syncthreads();

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;
			atomicAdd(result, sums[0]);
		}
		__syncthreads();

	}
}
__global__ void montecarloD1d(int N, float *result, curandState *state, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	__shared__ float sums[256];
	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandState lstate = state[id];
		float x = curand_uniform(&lstate);
		x *= (b-a);
		x+=a;
		//
		//printf("x = %g\t",x);
		sums[threadIdx.x] = func(x);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			//float s = sums[0] * (b - a) / 256;
			atomicAdd(result, sums[0]);
		}
		__syncthreads();
		//if(id == 0) *result *= (b - a) / N;
	}
}

//template<class OP>
__global__ void montecarloDnd(int N, float *result, curandState *state, int n, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sums[256];
	float vals[20];// = (float*)malloc(n*sizeof(float));
	sums[threadIdx.x] = 0.0f;

	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandState lstate = state[id];
		float x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&lstate);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			float s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}
__global__ void montecarloDndPhil(int N, float *result, curandStatePhilox4_32_10_t *state, int n, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sums[256];
	float vals[20];// = (float*)malloc(n*sizeof(float));
	sums[threadIdx.x] = 0.0f;

	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandStatePhilox4_32_10_t lstate = state[id];
		float x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&lstate);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			float s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}
__global__ void montecarloDndMrg(int N, float *result, curandStateMRG32k3a *state, int n, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sums[256];
	float vals[20];// = (float*)malloc(n*sizeof(float));
	sums[threadIdx.x] = 0.0f;

	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandStateMRG32k3a lstate = state[id];
		float x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&lstate);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			float s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}

__global__ void montecarloDndMt(int N, float *result, curandStateMtgp32_t *state, int n, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sums[256];
	float vals[20];
	sums[threadIdx.x] = 0.0f;



	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandStateMtgp32_t lstate = state[id];
		float x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&lstate);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			float s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}

__global__ void montecarloDndSobol(int N, float *result, curandStateSobol32 *state, int n, float a, float b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sums[256];
	float vals[20];// = (double*)malloc(n*sizeof(double));
	sums[threadIdx.x] = 0.0f;

	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		float x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&state[i + id*n]);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		//state[id] = lstate;

		if(threadIdx.x == 0){
			float s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}

__global__ void montecarloDndMrgDBL(int N, double *result, curandStateMRG32k3a *state, int n, double a, double b){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ double sums[256];
	double vals[20];
	sums[threadIdx.x] = 0.0f;



	if(id == 0){
		*result = 0;
	}
	__syncthreads();
	if(id < N){
		curandStateMRG32k3a lstate = state[id];
		double x = 1.0f / N;
		for(int i = 0; i < n; i++){
			vals[i] = curand_uniform(&lstate);
			vals[i] *= (b-a);
			vals[i] += a;
			x *= (b-a);
		}

		sums[threadIdx.x] = d_fop(vals);  // itt kell kiszámolni az aktuális értéket a random számmal

		//utána mehet a sum folding
		for(int i = blockDim.x / 2; i > 0; i /= 2){
			__syncthreads();
			if(threadIdx.x < i){
				sums[threadIdx.x] += sums[threadIdx.x + i];
			}
		}

		__syncthreads();
		state[id] = lstate;

		if(threadIdx.x == 0){
			double s = 1.0;
			for(int i = 0; i<n;i++) s *= (b-a);
			atomicAdd(result, s * sums[0] / N);

		}
		__syncthreads();

	}
}



