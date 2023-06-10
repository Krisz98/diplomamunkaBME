/*
 * convolutions.h
 *
 *  Created on: Mar 19, 2023
 *      Author: krisztian
 */

#ifndef CONVOLUTIONS_H_
#define CONVOLUTIONS_H_

#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 256
#define MASK_SIZE 1024

namespace Conv_ns{

extern __device__ __constant__ float G_MASK[MASK_SIZE];
extern __device__ float G_MASK_G[MASK_SIZE];



__global__ void conv1D1(const float* f, const int fn, float* y);
__global__ void conv1D1FMA(const float* f, const int fn, float* y);
__global__ void conv1D1F(const float* f, const int fn, float* y);
__global__ void conv1D1G(const float* f, const int fn, float* y);
__global__ void conv1D2(const float* f, const int fn, float* y);
void convHost(const float* f, const int fn, const float *g ,float *y);

}
#endif /* CONVOLUTIONS_H_ */
