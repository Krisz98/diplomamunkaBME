//============================================================================
// Name        : conv.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>
#include "cuda.h"
#include "cuda_runtime.h"
#include "convolutions.h"
#include <assert.h>
#include <cmath>
#include<vector>
#include <chrono>



bool checkResult(const float* actual, const float* expected, int n, float epsilon){
	for(int i = 0; i < n; i++){
		if(abs(actual[i] - expected[i]) > epsilon) return false;
	}
	return true;
}
void checkDifference(const float* actual, const float* expected, int n){
	std::cout<<"Difference: ";
	for(int i = 0; i < n; i++){
		std::cout<<expected[i] - actual[i]<<" | ";
	}
	std::cout<<std::endl;
	std::cout<<"Actual: ";
		for(int i = 0; i < n; i++){
			std::cout<<actual[i]<<" | ";
	}
	std::cout<<std::endl;
	std::cout<<"Expected: ";
	for(int i = 0; i < n; i++){
		std::cout<<expected[i]<<" | ";
	}
	std::cout<<std::endl<<std::endl;
}

void testConv1D1(){
	cudaError_t error;
	float *d_f, *d_y;
	int n = 10;

	error = cudaDeviceReset();
	assert(error == cudaSuccess);

	error = cudaMalloc(&d_f, n*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, n*sizeof(float));
	assert(error == cudaSuccess);
	//float mask[10] = {1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f};
	float mask[10] = {1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f};

	float f[10] = {1,1,1,1,1,1,1,1,1,1};
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK,mask, n *sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, n*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);


	Conv_ns::conv1D1<<<n/256 + 1, 256>>>(
				d_f,
				n,
				d_y);

	float* y = (float*)malloc(n*sizeof(float));
	error = cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
	assert(error == cudaSuccess);

	float y_exp[10] = {1.0f/10.0f, 2.0f/10.0f, 3.0f/10.0f, 4.0f/10.0f, 5.0f/10.0f, 6.0f/10.0f, 7.0f/10.0f, 8.0f/10.0f, 9.0f/10.0f, 10.0f/10.0f};
	checkDifference(y, y_exp, n);
	assert(checkResult(y, y_exp, n, 0.001f));
}
void testConv1D1FMA(){
	cudaError_t error;
	float *d_f, *d_y;
	int n = 10;

	error = cudaDeviceReset();
	assert(error == cudaSuccess);

	error = cudaMalloc(&d_f, n*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, n*sizeof(float));
	assert(error == cudaSuccess);
	//float mask[10] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
	//float mask[10] = {1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f};
	float mask[10] = {1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f,1.0f/10.0f};

	float f[10] = {1,1,1,1,1,1,1,1,1,1};
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK,mask, n *sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, n*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);


	Conv_ns::conv1D1FMA<<<n/256 + 1, 256>>>(
				d_f,
				n,
				d_y);

	float* y = (float*)malloc(n*sizeof(float));
	error = cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
	assert(error == cudaSuccess);

	//float y_exp[10] = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};
	float y_exp[10] = {1.0f/10.0f, 2.0f/10.0f, 3.0f/10.0f, 4.0f/10.0f, 5.0f/10.0f, 6.0f/10.0f, 7.0f/10.0f, 8.0f/10.0f, 9.0f/10.0f, 10.0f/10.0f};

	checkDifference(y, y_exp, n);
	assert(checkResult(y, y_exp, n, 0.001f));
}
void testConv1D1G(){
	cudaError_t error;
	float *d_f, *d_y;
	int n = 10;

	error = cudaDeviceReset();
	assert(error == cudaSuccess);

	error = cudaMalloc(&d_f, n*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, n*sizeof(float));
	assert(error == cudaSuccess);
	float mask[10] = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
	float f[10] = {1,1,1,1,1,1,1,1,1,1};
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK_G,mask, n *sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, n*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);


	Conv_ns::conv1D1G<<<n/256 + 1, 256>>>(
				d_f,
				n,
				d_y);

	float* y = (float*)malloc(n*sizeof(float));
	error = cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
	assert(error == cudaSuccess);

	float y_exp[10] = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};
	assert(checkResult(y, y_exp, n, 0.001f));
}

float measureConv1D1(const float* f, const int N, float* y, float* mask, float* meas_time, const int n_measures, float &avg){

	cudaError_t error;
	cudaEvent_t start, stop;
	error = cudaEventCreate(&start);
	assert(error == cudaSuccess);
	error = cudaEventCreate(&stop);
	assert(error == cudaSuccess);

	float *d_f, *d_y;
	avg = 0.0f;

	error = cudaMalloc(&d_f, N*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, N*sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, N*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);
	//error = cudaGetSymbolAddress((void **)&d_g,(const void*)G_MASK2);
	//error = cudaMemcpy(d_g, mask, N*sizeof(float), cudaMemcpyHostToDevice);
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK, mask, N *sizeof(float));
	assert(error == cudaSuccess);

	for(int i = 0; i < n_measures; i++){
		error = cudaEventRecord(start);
		assert(error == cudaSuccess);

		Conv_ns::conv1D1<<<N/256 + 1, 256>>>(
				d_f,
				N,
				d_y);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			std::cout << "Error: "<< cudaGetErrorName(error)<<" --> " << cudaGetErrorString(error) << std::endl;
		}
		error = cudaEventRecord(stop);

		assert(error == cudaSuccess);

		error = cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
		assert(error == cudaSuccess);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		meas_time[i] = milliseconds;
		std::cout<< "Elapsed time: "<<milliseconds<< "ms"<<std::endl;

		avg += milliseconds / n_measures;

	}
	cudaFree(d_f);
	cudaFree(d_y);

	return 0;
}
float measureConv1D1FMA(const float* f, const int N, float* y, float* mask, float* meas_time, const int n_measures, float &avg){

	cudaError_t error;
	cudaEvent_t start, stop;
	error = cudaEventCreate(&start);
	assert(error == cudaSuccess);
	error = cudaEventCreate(&stop);
	assert(error == cudaSuccess);

	float *d_f, *d_y;
	avg = 0.0f;

	error = cudaMalloc(&d_f, N*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, N*sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, N*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);
	//error = cudaGetSymbolAddress((void **)&d_g,(const void*)G_MASK2);
	//error = cudaMemcpy(d_g, mask, N*sizeof(float), cudaMemcpyHostToDevice);
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK, mask, N *sizeof(float));
	assert(error == cudaSuccess);

	for(int i = 0; i < n_measures; i++){
		error = cudaEventRecord(start);
		assert(error == cudaSuccess);

		Conv_ns::conv1D1FMA<<<N/256 + 1, 256>>>(
				d_f,
				N,
				d_y);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			std::cout << "Error: "<< cudaGetErrorName(error)<<" --> " << cudaGetErrorString(error) << std::endl;
		}
		error = cudaEventRecord(stop);

		assert(error == cudaSuccess);

		error = cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
		assert(error == cudaSuccess);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		meas_time[i] = milliseconds;
		std::cout<< "Elapsed time: "<<milliseconds<< "ms"<<std::endl;

		avg += milliseconds / n_measures;

	}
	cudaFree(d_f);
	cudaFree(d_y);

	return 0;
}
float measureConv1D1G(const float* f, const int N, float* y, float* mask, float* meas_time, const int n_measures, float &avg){

	cudaError_t error;
	cudaEvent_t start, stop;
	error = cudaEventCreate(&start);
	assert(error == cudaSuccess);
	error = cudaEventCreate(&stop);
	assert(error == cudaSuccess);

	float *d_f, *d_y;
	avg = 0.0f;

	error = cudaMalloc(&d_f, N*sizeof(float));
	assert(error == cudaSuccess);
	cudaMalloc(&d_y, N*sizeof(float));
	assert(error == cudaSuccess);

	error = cudaMemcpy(d_f, f, N*sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);
	error = cudaMemcpyToSymbol(Conv_ns::G_MASK_G,mask, N *sizeof(float));
	assert(error == cudaSuccess);

	for(int i = 0; i < n_measures; i++){
		error = cudaEventRecord(start);
		assert(error == cudaSuccess);

		Conv_ns::conv1D1G<<<N/256 + 1, 256>>>(
				d_f,
				N,
				d_y);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			std::cout << "Error: "<< cudaGetErrorName(error)<<" --> " << cudaGetErrorString(error) << std::endl;
		}
		error = cudaEventRecord(stop);

		assert(error == cudaSuccess);

		error = cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
		assert(error == cudaSuccess);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		meas_time[i] = milliseconds;
		std::cout<< "Elapsed time: "<<milliseconds<< "ms"<<std::endl;

		avg += milliseconds / n_measures;

	}
	cudaFree(d_f);
	cudaFree(d_y);

	return 0;
}
float measureConvHost(const float* f, const int N, float* y, float* mask, float* meas_time, const int n_measures, float &avg){

	float *d_f, *d_y;
	avg = 0.0f;

	for(int i = 0; i < n_measures; i++){
		auto start = std::chrono::steady_clock::now();
		Conv_ns::convHost(f, N, mask, y);
		auto end = std::chrono::steady_clock::now();

		double milliseconds = 0;
		milliseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
		meas_time[i] = milliseconds / 1000000.0f;
		std::cout<< "Elapsed time: "<<milliseconds / 1000.0f<< "ms"<<std::endl;

		avg += milliseconds / n_measures / 1000000.0f;

	}
	cudaFree(d_f);
	cudaFree(d_y);

	return 0;
}

void initarray(float* array, const int n, float val){
	for(int i = 0; i < n; i++){
		array[i] = val;
	}
}

int main() {
	std::cout << std::setprecision(32);
	testConv1D1();
	testConv1D1FMA();
	const int N_measurements = 10;
	float time_measurements[N_measurements];
	float avg;
	for(int i = 0; i < 5; i++){
		int n = 200*i;
		assert(n <= MASK_SIZE);
		// memóriát allokálok a bemenetnek:
		float* h_f = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a maszknak
		float *h_g = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a kimenetnek
		float *h_y = (float*) malloc(n * sizeof(float));

		initarray(h_f, n, 1.5);
		initarray(h_g, n, 1.2);

		measureConv1D1(h_f, n, h_y, h_g, time_measurements, N_measurements, avg);

		std::cout<<"Measurement 1D1 "<<i<<". | #of measurements: "<<N_measurements<<" | series length: "<<n<<" | avg: "<<avg<<"ms"<<std::endl;

		free(h_y);
		free(h_g);
		free(h_f);
	}
	for(int i = 0; i < 5; i++){
			int n = 200*i;
			assert(n <= MASK_SIZE);
			// memóriát allokálok a bemenetnek:
			float* h_f = (float*) malloc(n*sizeof(float));
			// memóriát allokálok a maszknak
			float *h_g = (float*) malloc(n*sizeof(float));
			// memóriát allokálok a kimenetnek
			float *h_y = (float*) malloc(n * sizeof(float));

			initarray(h_f, n, 1.5);
			initarray(h_g, n, 1.2);

			measureConv1D1(h_f, n, h_y, h_g, time_measurements, N_measurements, avg);

			std::cout<<"Measurement 1D1 "<<i<<". | #of measurements: "<<N_measurements<<" | series length: "<<n<<" | avg: "<<avg<<"ms"<<std::endl;

			free(h_y);
			free(h_g);
			free(h_f);
		}

	testConv1D1G();

	for(int i = 0; i < 5; i++){
		int n = 200*i;
		assert(n <= MASK_SIZE);
		// memóriát allokálok a bemenetnek:
		float* h_f = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a maszknak
		float *h_g = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a kimenetnek
		float *h_y = (float*) malloc(n * sizeof(float));

		initarray(h_f, n, 1.5);
		initarray(h_g, n, 1.2);

		measureConv1D1G(h_f, n, h_y, h_g, time_measurements, N_measurements, avg);
		std::cout<<"Measurement 1D1G "<<i<<". | #of measurements: "<<N_measurements<<" | series length: "<<n<<" | avg: "<<avg <<"ms"<<std::endl;
		free(h_y);
		free(h_g);
		free(h_f);
	}
	for(int i = 0; i < 5; i++){
		int n = 200*i;
		assert(n <= MASK_SIZE);
		// memóriát allokálok a bemenetnek:
		float* h_f = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a maszknak
		float *h_g = (float*) malloc(n*sizeof(float));
		// memóriát allokálok a kimenetnek
		float *h_y = (float*) malloc(n * sizeof(float));

		initarray(h_f, n, 1.5);
		initarray(h_g, n, 1.2);

		measureConvHost(h_f, n, h_y, h_g, time_measurements, N_measurements, avg);
		std::cout<<"Measurement HOST "<<i<<". | #of measurements: "<<N_measurements<<" | series length: "<<n<<" | avg: "<<avg<<"ms"<<std::endl;
		free(h_y);
		free(h_g);
		free(h_f);
	}
	

	return 0;
}



