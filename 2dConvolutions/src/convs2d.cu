/*
 * convs2d.cu
 *
 *  Created on: Apr 3, 2023
 *      Author: krisztian
 */

#include "convs2d.h"

__device__ float Conv2d::kernel[KERNEL_WIDTH *  KERNEL_HEIGHT];
__device__ float Conv2d::kernel_horizontal[KERNEL_WIDTH];
__device__ float Conv2d::kernel_vertical[KERNEL_HEIGHT];

__global__ void Conv2d::c2dG(unsigned char * volatile data, unsigned char* result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int ga = x + y * width;
	float sum = 0.0f;

	if(x < width && y < height){
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			int py = y - (int)(KERNEL_HEIGHT / 2) + i;
			for(int j = 0; j < KERNEL_WIDTH; j++){
				int px = x - (int)(KERNEL_WIDTH / 2) + j;

				float pv = (px < 0 || px > width || py < 0 || py > height) ? 0.0f : (float)data[px + py * width];
				//if((px > 0 && px < width && py > 0 && py < height))data[px + py * width] += 0;
				float kv = Conv2d::kernel[j + i * KERNEL_WIDTH];

				sum += kv * pv;
			}
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}
}

__global__ void Conv2d::c2dS(unsigned char * volatile data, unsigned char *result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	__shared__ unsigned char blockPixels[4 * BLOCK_SIZE * BLOCK_SIZE];

	if(x < width && y < height){
		{
			int Ix0 = x - threadIdx.x;  // pixel globális koordinátája
			int Iy0 = y - threadIdx.y;
			int Ax0 = Ix0 - BLOCK_SIZE / 2; // keret globális koordinátája->lehet negatív is a széleken
			int Ay0 = Iy0 - BLOCK_SIZE / 2;
			int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

			// Betöltöm a pixeleket
			for (int i = 0; i<4;i++){
				int a = bid *4 + i;
				int ay = a / (2 * BLOCK_SIZE);
				int ax = a - ay * 2 * BLOCK_SIZE;
				int Ax = Ax0 + ax;
				int Ay = Ay0 + ay;
				int Aidx = Ax + Ay * width;
				blockPixels[a] = (Ax > 0 && Ay > 0 && Ax < width && Ay < height) ? data[Aidx] : 0;
			}
		}
		__syncthreads();
		int pax = threadIdx.x + BLOCK_SIZE / 2;  // pixel koordináta a lokális koordináta rendszerben
		int pay = threadIdx.y + BLOCK_SIZE / 2;
		float sum = 0.0f;
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			int ay = pay - (int)(KERNEL_HEIGHT / 2) + i;
			for(int j = 0; j < KERNEL_WIDTH; j++){
				int ax = pax - (int)(KERNEL_WIDTH / 2) + j;
				int a = ax + ay * 2 * BLOCK_SIZE;

				float kv = Conv2d::kernel[j + i * KERNEL_WIDTH];
				sum += (float)blockPixels[a] * kv;
			}
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}
}

__global__ void Conv2d::c2dS2(unsigned char * volatile data, unsigned char *result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	__shared__ unsigned char blockPixels[4 * BLOCK_SIZE * BLOCK_SIZE];

	if(x < width && y < height){
		{
			int Ix0 = x - threadIdx.x;  // pixel globális koordinátája
			int Iy0 = y - threadIdx.y;
			int Ax0 = Ix0 - KERNEL_WIDTH / 2; // keret globális koordinátája->lehet negatív is a széleken
			int Ay0 = Iy0 - KERNEL_HEIGHT / 2;
			int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

			int n = (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2) * (BLOCK_SIZE + KERNEL_HEIGHT / 2 * 2) / BLOCK_SIZE / BLOCK_SIZE + 1;

			// Betöltöm a pixeleket
			for (int i = 0; i<n;i++){
				int a = bid *n + i;
				int ay = a / (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2);
				int ax = a - ay * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2);
				int Ax = Ax0 + ax;
				int Ay = Ay0 + ay;
				int Aidx = Ax + Ay * width;
				blockPixels[a] = (Ax > 0 && Ay > 0 && Ax < width && Ay < height) ? data[Aidx] : 0;
			}
		}
		__syncthreads();
		int pax = threadIdx.x + (KERNEL_WIDTH / 2);  // pixel koordináta a lokális koordináta rendszerben
		int pay = threadIdx.y + (KERNEL_HEIGHT / 2);
		float sum = 0.0f;
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			int ay = pay - (int)(KERNEL_HEIGHT / 2) + i;
			for(int j = 0; j < KERNEL_WIDTH; j++){
				int ax = pax - (int)(KERNEL_WIDTH / 2) + j;
				int a = ax + ay * (BLOCK_SIZE + KERNEL_WIDTH / 2 * 2);

				float kv = Conv2d::kernel[j + i * KERNEL_WIDTH];
				sum += (float)blockPixels[a] * kv;
			}
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}
}

__global__ void Conv2d::c2dSHorizontal(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	const int tsize = BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE * KERNEL_WIDTH / 2;

	__shared__ unsigned char blockPixels[tsize];


	if(x < width && y < height){
		int nload = tsize / (blockDim.x * blockDim.y) + 1;
		int I0 = ga - threadIdx.x;

		int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

		const int row_length = KERNEL_WIDTH / 2 * 2 + BLOCK_SIZE;

		for(int i = 0; i < nload; i++){
			int aindex = bid * nload + i;
			int ay = aindex / row_length;
			int ax = aindex - ay * row_length;
			int Ax = x - threadIdx.x - (KERNEL_WIDTH / 2) + ax;
			int Ay = y - threadIdx.y + ay;
			blockPixels[aindex] = (Ax < 0 || Ax > width) ? 0 : data[Ax + width * Ay];
		}
		__syncthreads();

		result[ga] = blockPixels[threadIdx.x + KERNEL_WIDTH / 2 + row_length * threadIdx.y];

		float sum = 0.0f;
		int pax = threadIdx.x + KERNEL_WIDTH / 2;
		for(int i = 0; i < KERNEL_WIDTH; i++){
			sum += blockPixels[pax - KERNEL_WIDTH / 2 + i + threadIdx.y * row_length] * Conv2d::kernel_horizontal[i];
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}

}

__global__ void Conv2d::c2dSVertical(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	const int tsize = BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE * KERNEL_HEIGHT / 2;

	__shared__ unsigned char blockPixels[tsize];

	if(x < width && y < height){
		int nload = tsize / (blockDim.x * blockDim.y) + 1;
		int I0 = ga - threadIdx.x;

		int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

		const int col_length = KERNEL_HEIGHT / 2 * 2 + BLOCK_SIZE;
		const int row_length = BLOCK_SIZE;

		for(int i = 0; i < nload; i++){
			int aindex = bid *nload + i;
			int ay = aindex / row_length;
			int ax = aindex - ay * row_length;
			int Ax = x - threadIdx.x + ax;
			int Ay = y - threadIdx.y - (KERNEL_HEIGHT / 2) + ay;
			blockPixels[aindex] = (Ay < 0 || Ay > height) ? 0 : data[Ax + width * Ay];
		}
		__syncthreads();
		float sum = 0.0f;
		int pax = threadIdx.x;
		int pay = threadIdx.y + KERNEL_HEIGHT / 2;
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			sum += blockPixels[pax + (pay - KERNEL_HEIGHT / 2 + i)*row_length ] * Conv2d::kernel_vertical[i];
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}


}
__global__ void Conv2d::c2dSHorizontal_fused(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	const int tsize = BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE * KERNEL_WIDTH / 2;

	__shared__ unsigned char blockPixels[tsize];


	if(x < width && y < height){
		int nload = tsize / (blockDim.x * blockDim.y) + 1;
		int I0 = ga - threadIdx.x;

		int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

		const int row_length = KERNEL_WIDTH / 2 * 2 + BLOCK_SIZE;

		for(int i = 0; i < nload; i++){
			int aindex = bid * nload + i;
			int ay = aindex / row_length;
			int ax = aindex - ay * row_length;
			int Ax = x - threadIdx.x - (KERNEL_WIDTH / 2) + ax;
			int Ay = y - threadIdx.y + ay;
			blockPixels[aindex] = (Ax < 0 || Ax > width) ? 0 : data[Ax + width * Ay];
		}
		__syncthreads();

		result[ga] = blockPixels[threadIdx.x + KERNEL_WIDTH / 2 + row_length * threadIdx.y];

		float sum = 0.0f;
		int pax = threadIdx.x + KERNEL_WIDTH / 2;
		for(int i = 0; i < KERNEL_WIDTH; i++){
			sum = __fma_rn(blockPixels[pax - KERNEL_WIDTH / 2 + i + threadIdx.y * row_length], Conv2d::kernel_horizontal[i], sum);
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}

}
__global__ void Conv2d::c2dSVertical_fused(unsigned char *data, unsigned char* result, unsigned int width, unsigned int height){
	int x = threadIdx.x + blockIdx.x * blockDim.x;  // Ix
	int y = threadIdx.y + blockIdx.y * blockDim.y;  // Iy
	int ga = x + y * width;

	const int tsize = BLOCK_SIZE * BLOCK_SIZE + 2 * BLOCK_SIZE * KERNEL_HEIGHT / 2;

	__shared__ unsigned char blockPixels[tsize];

	if(x < width && y < height){
		int nload = tsize / (blockDim.x * blockDim.y) + 1;
		int I0 = ga - threadIdx.x;

		int bid = threadIdx.x + blockDim.x * threadIdx.y; // id in block

		const int col_length = KERNEL_HEIGHT / 2 * 2 + BLOCK_SIZE;
		const int row_length = BLOCK_SIZE;

		for(int i = 0; i < nload; i++){
			int aindex = bid *nload + i;
			int ay = aindex / row_length;
			int ax = aindex - ay * row_length;
			int Ax = x - threadIdx.x + ax;
			int Ay = y - threadIdx.y - (KERNEL_HEIGHT / 2) + ay;
			blockPixels[aindex] = (Ay < 0 || Ay > height) ? 0 : data[Ax + width * Ay];
		}
		__syncthreads();
		float sum = 0.0f;
		int pax = threadIdx.x;
		int pay = threadIdx.y + KERNEL_HEIGHT / 2;
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			sum = __fma_rn(blockPixels[pax + (pay - KERNEL_HEIGHT / 2 + i)*row_length ], Conv2d::kernel_vertical[i], sum);
		}
		if(sum > 255.0f) result[ga] = (unsigned char)255;
		else if(sum < 0) result[ga] = (unsigned char)0;
		else result[ga] = (unsigned char)sum;
	}
}




