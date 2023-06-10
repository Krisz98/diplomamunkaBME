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
void generateArr(float* data, int n, int a, int b){
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


void printarr(float *data, int l, int h){
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
	cout<<"duration: "<<dt.count() / 1000000.0f<<"ms"<<endl;

}

__host__ __device__
void merge(float* data, float* result, int li, int w, int n){
	float *p1 = data + li;
	float *p2 = data + li + w;
	float *r = result + li;

	int i = 0, j = 0, k = 0;
	while(i < w && j < w && (li + i) < n && (li + w + j) < n){
		if(p1[i] < p2[j]){
			r[k++] = p1[i++];
		}
		else{
			r[k++] = p2[j++];
		}
	}
	while(i < w && (li + i) < n) r[k++] = p1[i++];
	while(j < w && (li + w + j) < n) r[k++] = p2[j++];
}

__global__ void d_mergesort(float *data, float* result, int N){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int n0 = ((N%2)==0) ? N/2 : N/2 +1;
	unsigned int w;
	int i;
	int k = 0;
	for(w = 1; w < N; w = w<<1){
		i = w<<1;
		n0 = ((N%i)==0) ? N/i : N/i +1;
		if(idx < n0){
			int l = idx*i;
			if(k == 0) merge(data, result, l, w, N);
			else merge(result, data, l, w, N);
		}
		k ^=1;
		//__syncthreads();
	}
}

__global__ void d_mergesortshared(float *data, float* result, int N){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int n0;
	__shared__ float sdata[256], sresult[256];
	if(idx < N)
	{
		sdata[threadIdx.x] = data[idx];
		__syncthreads();

		unsigned int w;
		int i;
		int k = 0;
		int s = min(N - blockIdx.x * 256, 256);
		#pragma unroll
		for(w = 1; w < 256; w = w<<1){
			i = w<<1;
			n0 = 256 / i;
			if(threadIdx.x < n0){
				int l = threadIdx.x*i;
				if(k == 0) merge(sdata, sresult, l, w, s);
				else merge(sresult, sdata, l, w, s);

			}
			k ^=1;
			__syncthreads();
		}
		data[idx] = sdata[threadIdx.x];
		result[idx] = sresult[threadIdx.x];
		__syncthreads();
		#pragma unroll
		for(w = 256; w < N; w = w<<1){
			i = w<<1;
			n0 = ((N%i)==0) ? N/i : N/i +1;
			if(idx < n0){
				int l = idx*i;
				if(k == 0) merge(data, result, l, w, N);
				else merge(result, data, l, w, N);

			}
			k ^=1;
			__syncthreads();
		}

	}
}



void testmergesort_d(int n_meas, int length){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	//float data[] = {1, 33, 10, 4, 8, 2, 5};
	float *data;
	//float data[] =  {10, 7, 8, 9, 1, 5 };
	int n = length;// sizeof(data) / sizeof(float);
	data = (float*)malloc(n*sizeof(float));

	float *data_d, *result_d;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&data_d, n*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&result_d, n*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(data_d, &data[0], n*sizeof(float), cudaMemcpyHostToDevice));

	float *ets;
	ets = (float*)malloc(n_meas*sizeof(float));



	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(256);
	dim3 blocksPerGrid(n/256 + 1);
	float et;
	cudaError_t e;

	//float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		generateArr(data, n, 0, 1000);
		//cout<<"Original array"<<endl;
		//printarr(data, 0, n-1);
		CHECK_CUDA_ERROR(cudaMemcpy(data_d, data, n*sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		d_mergesort<<<blocksPerGrid,threadsPerBlock >>>(data_d, result_d, n);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		int d = log2(n);
		if(n - pow(2, d) != 0) d++;
		//cout<<"D = "<<d<<endl;
		if(d % 2 == 0)CHECK_CUDA_ERROR(cudaMemcpy(data, data_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		else CHECK_CUDA_ERROR(cudaMemcpy(data, result_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		//printarr(data, 0, n-1);
		//else CHECK_CUDA_ERROR(cudaMemcpy(data, result_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Mergesort_d  iteration: "<<i<<"\t | "<< "elapsed time: "<<et<<"ms"<<endl;
		//avg += et / n_meas;
		//cout<<"Result:"<<endl;


		ets[i] = et;
	}
	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	//float sdet = calcSTD(ets, n_meas, avget);
	//cout<< "SD_et: "<<sdet<<"ms"<<endl;



	CHECK_CUDA_ERROR(cudaFree(data_d));
	CHECK_CUDA_ERROR(cudaFree(result_d));
	free(ets);
	free(data);
	cudaDeviceSynchronize();
}

void testmergesort_d_shared(int n_meas, int length){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));

	//float data[] = {1, 33, 10, 4, 8, 2, 5};
	float *data;
	//float data[] =  {10, 7, 8, 9, 1, 5 };
	int n = length;// sizeof(data) / sizeof(float);
	data = (float*)malloc(n*sizeof(float));

	float *data_d, *result_d;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&data_d, n*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&result_d, n*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(data_d, &data[0], n*sizeof(float), cudaMemcpyHostToDevice));

	float *ets;
	ets = (float*)malloc(n_meas*sizeof(float));



	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(256);
	dim3 blocksPerGrid(n/256 + 1);
	float et;
	cudaError_t e;

	//float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		generateArr(data, n, 0, 1000);
		//cout<<"Original array"<<endl;
		//printarr(data, 0, n-1);
		CHECK_CUDA_ERROR(cudaMemcpy(data_d, data, n*sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		d_mergesortshared<<<blocksPerGrid,threadsPerBlock >>>(data_d, result_d, n);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		int d = log2(n);
		if(n - pow(2, d) != 0) d++;
		//cout<<"D = "<<d<<endl;
		if(d % 2 == 0)CHECK_CUDA_ERROR(cudaMemcpy(data, data_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		else CHECK_CUDA_ERROR(cudaMemcpy(data, result_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		//printarr(data, 0, n-1);
		//else CHECK_CUDA_ERROR(cudaMemcpy(data, result_d, n*sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Mergesort_d  iteration: "<<i<<"\t | "<< "elapsed time: "<<et<<"ms"<<endl;
		//avg += et / n_meas;
		//cout<<"Result:"<<endl;
		ets[i] = et;
	}
	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	//float sdet = calcSTD(ets, n_meas, avget);
	//cout<< "SD_et: "<<sdet<<"ms"<<endl;



	CHECK_CUDA_ERROR(cudaFree(data_d));
	CHECK_CUDA_ERROR(cudaFree(result_d));
	free(ets);
	free(data);

	cudaDeviceSynchronize();
}


int main(){
	int n = 1000000;
	cout<<"CPU_quicksort: "<<endl;
	testquicksort(n);
	cout<<endl<<"GPU_mergesort: "<<endl;
	testmergesort_d(10, n);
	cout<<endl<<"GPU_mergesort_shared: "<<endl;
	testmergesort_d_shared(10, n);

	float data[] = {3,4,1,2};
	float res[4];
	merge(data, res, 0, 2, 4);
	printarr(res, 0, 3);
}




