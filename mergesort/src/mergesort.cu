//============================================================================
// Name        : mergesort.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

/*
 * main.cpp
 *
 *  Created on: May 24, 2023
 *      Author: krisztian
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>
#include "curand_kernel.h"
#include <cuda.h>
#include "cuda_runtime.h"

using namespace std;

#define DEPTH_MAX 24

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

template<typename T>
void generateArr(T* data, int n, int a, int b){
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(a,b);
	auto gen = std::bind(distribution, generator);
	for(int i = 0; i < n; i++) data[i] = gen();
}

template<typename T>
float calcavg(T* data, int n){
	double sum = 0.0;
	for(int i = 0; i < n; i++){
		sum += data[i];
	}
	return (T) (sum / n);
}


template<typename T>
void printarr(T *data, int l, int h){
	for(int i = l; i <= h; i++){
		cout<<data[i]<<"; ";
	}
	cout<<endl;
}

__host__ __device__ int partition(float* data, int li, int ri){
	int pi = ri;
	int si = li - 1;
	for (int i = li; i <= ri-1; i++){
		if(data[i] < data[pi]){
			si++;
			float tmp = data[i];
			data[i] = data[si];
			data[si] = tmp;
		}
	}
	float tmp = data[pi];
	data[pi] = data[si + 1];
	data[si + 1] = tmp;
	return si + 1;
}

template<typename T>
__device__ int partitionGPU(T* data, int li, int ri){
	int pi = ri;
	int si = li - 1;
	for (int i = li; i <= ri-1; i++){
		if(data[i] < data[pi]){
			si++;
			T tmp = data[i];
			data[i] = data[si];
			data[si] = tmp;
		}
	}
	T tmp = data[pi];
	data[pi] = data[si + 1];
	data[si + 1] = tmp;
	return si + 1;
}

void quicksort(float *data, int li, int ri){
	if(li < ri){
		int pi = partition(data, li, ri);
		quicksort(data, li, pi - 1);
		quicksort(data, pi+1, ri);
	}
}



void testquicksort(int length){
	float *data;
	int n = length;// sizeof(data) / sizeof(float);
	data = (float*)malloc(n*sizeof(float));
	generateArr(data, n, 0, 1000
			);

	//printarr(data,0,n-1);
	auto start = std::chrono::steady_clock::now();
	quicksort(data, 0, n-1);
	auto end = std::chrono::steady_clock::now();
	//printarr(data, 0,n-1);
	auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
	cout<<"duration: "<<dt.count()/1000000.0<<"ms"<<endl;

}

template<typename T>
__host__ __device__
void selection_sort(T *data, int left, int right) {
  for (int i = left; i <= right; ++i) {
    T min_val = data[i];
    int min_idx = i;

    for (int j = i + 1; j <= right; ++j) {
      T val_j = data[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void d_quicksort(float *data, int li, int ri){
	if(li < ri){
			int pi = partitionGPU(data, li, ri);
			dim3 bpg(1);
			dim3 tpb(1);
			d_quicksort<<<bpg,tpb>>>(data, li, pi - 1);
			d_quicksort<<<bpg,tpb>>>(data, pi+1, ri);


	}
}
__global__ void d_quicksort2(float *data, int li, int ri){
	if(li < ri){
			int pi = partitionGPU(data, li, ri);
			dim3 bpg(1);
			dim3 tpb(1);
			d_quicksort2<<<bpg,tpb>>>(data, li, pi - 1);
			d_quicksort2<<<bpg,tpb>>>(data, pi+1, ri);
			//cudaDeviceSynchronize();

	}
}

template<typename T>
__global__ void d_quicksort3(T *data, int li, int ri, int depth){
	if(li < ri){
		if(depth > DEPTH_MAX){
			selection_sort(data, li, ri);
			return;
		}
		int pi = partitionGPU(data, li, ri);
		dim3 bpg(1);
		dim3 tpb(1);
		cudaStream_t s1, s2;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		d_quicksort3<<<bpg,tpb, 0, s1>>>(data, li, pi - 1, ++depth);
		d_quicksort3<<<bpg,tpb, 0, s2>>>(data, pi+1, ri, depth);
		cudaStreamDestroy(s1);
		cudaStreamDestroy(s2);

	}
}


template<typename T>
void testquicksort_d(int n_meas, int length){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	//float data[] = {1, 33, 10, 4, 8, 2, 5};
	T *data;
	//float data[] =  {10, 7, 8, 9, 1, 5 };
	int n = length;// sizeof(data) / sizeof(float);
	data = (T*)malloc(n*sizeof(T));

	T *data_d;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&data_d, n*sizeof(T)));
	CHECK_CUDA_ERROR(cudaMemcpy(data_d, &data[0], n*sizeof(T), cudaMemcpyHostToDevice));

	float *ets;
	ets = (float*)malloc(n_meas*sizeof(float));



	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(1);
	dim3 blocksPerGrid(1);
	float et;
	cudaError_t e;

	//float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		generateArr(data, n, 0, 1000);
		//cout<<"Original array"<<endl;
		//printarr(data, 0, n-1);
		CHECK_CUDA_ERROR(cudaMemcpy(data_d, data, n*sizeof(T), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768<<2);
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		d_quicksort3<<<blocksPerGrid,threadsPerBlock >>>(data_d, 0, n-1,0);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(data, data_d, n*sizeof(T), cudaMemcpyDeviceToHost));
		cout<<"Quicksort_d iteration: "<<i<<"\t | "<< "elapsed time: "<<et<<"ms"<<endl;
		//avg += et / n_meas;
		//cout<<"Result:"<<endl;
		//printarr(data, 0, n-1);

		ets[i] = et;
	}
	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	//float sdet = calcSTD(ets, n_meas, avget);
	//cout<< "SD_et: "<<sdet<<"ms"<<endl;



	CHECK_CUDA_ERROR(cudaFree(data_d));
	cudaDeviceSynchronize();
}



int main(){
	int n = 1000000;
	cout<<"CPU: "<<endl;
	testquicksort(n);
	cout<<"GPU: "<<endl;
	testquicksort_d<int>(10, n);
}





