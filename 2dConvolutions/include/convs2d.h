/*
 * convs2d.h
 *
 *  Created on: Apr 3, 2023
 *      Author: krisztian
 */

#ifndef CONVS2D_H_
#define CONVS2D_H_

#include <cuda.h>
#include "cuda_runtime.h"

namespace Conv2d{

#define BLOCK_SIZE 16
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3

#define EDGE_KERNEL {-1.0f, -1.0f, -1.0f,\
					-1.0f, 9.0f, -1.0f,\
					-1.0f, -1.0f, -1.0f}
#define SHARPEN_KERNEL {0.0f, -1.0f, 0.0f,\
						-1.0f, 5.0f, -1.0f,\
						0.0f, -1.0f, 0.0f}
#define BOXBLUR_KERNEL {1/9.0f, 1/9.0f, 1/9.0f,\
						1/9.0f, 1/9.0f, 1/9.0f,\
						1/9.0f, 1/9.0f, 1/9.0f}
#define IDENTITY_KERNEL {0.0f, 0.0f, 0.0f,\
						 0.0f, 1.0f, 0.0f,\
						 0.0f, 0.0f, 0.0f}

#define TEST_15_IDENT_KERNEL {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,\
						   0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}

#define HORIZONTAL_IDENTITY_KERNEL {0.0f, 1.0f, 0.0f}
#define VERTICAL_IDENTITY_KERNEL {0.0f, 1.0f, 0.0f}

#define HORIZONTAL_IDENTITY_KERNEL_15 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
#define VERTICAL_IDENTITY_KERNEL_15 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
extern __device__  float kernel[KERNEL_WIDTH *  KERNEL_HEIGHT];
extern __device__ float kernel_horizontal[KERNEL_WIDTH];
extern __device__ float kernel_vertical[KERNEL_HEIGHT];

__global__ void c2dG(unsigned char * volatile data, unsigned char* result, unsigned int width, unsigned int height);
__global__ void c2dS(unsigned char * volatile  data, unsigned char* result, unsigned int width, unsigned int height);

__global__ void c2dS2(unsigned char * volatile data, unsigned char* result, unsigned int width, unsigned int height);

__global__ void c2dSHorizontal(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height);
__global__ void c2dSHorizontal_fused(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height);
__global__ void c2dSVertical(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height);
__global__ void c2dSVertical_fused(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height);
}



#endif /* CONVS2D_H_ */
