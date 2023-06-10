/*
 * mc.h
 *
 *  Created on: May 4, 2023
 *      Author: krisztian
 */

#ifndef MC_H_
#define MC_H_

#include <cuda.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"


__global__ void setupmon(curandState *state, int N, int *j);
__global__ void setupmonPhil(curandStatePhilox4_32_10_t *state, int N, int *j);
__global__ void setupmonMrg(curandStateMRG32k3a *state, int N, int *j);
__global__ void setupmonSobol(curandStateSobol32 *state, unsigned int *direction_vectors, int N, int *j, int ndim);
__global__ void midpointrule(int N, float *result, float a, float b);
__global__ void midpointruleFMA(int N, float *result, float a, float b);
__global__ void trapezoidrule(int N, float *result, float a, float b);
__global__ void simpsonsrule(int N, float *result, float a, float b);
__global__ void simpsonsruleFMA(int N, float *result, float a, float b);

__global__ void montecarloD1d(int N, float *result, curandState *state, float a, float b);


__global__ void montecarloDnd(int N, float *result, curandState *state, int n, float a, float b);
__global__ void montecarloDndPhil(int N, float *result, curandStatePhilox4_32_10_t *state, int n, float a, float b);
__global__ void montecarloDndMrg(int N, float *result, curandStateMRG32k3a *state, int n, float a, float b);
__global__ void montecarloDndMt(int N, float *result, curandStateMtgp32_t *state, int n, float a, float b);
__global__ void montecarloDndSobol(int N, float *result, curandStateSobol32 *state, int n, float a, float b);
__global__ void montecarloDndMrgDBL(int N, double *result, curandStateMRG32k3a *state, int n, double a, double b);
__device__ float fop(float  *vs);
__device__ float d_fop(double  *vs);

__device__ float func(float x);
__device__ float funcf(float x);

#endif /* MC_H_ */
