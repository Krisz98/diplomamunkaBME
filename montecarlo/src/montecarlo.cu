//============================================================================
// Name        : montecarlo.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>
#include "mc.h"
#include "curand_kernel.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <curand.h>
#include "curand_globals.h"


using namespace std;

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
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

template<typename T>
float calcavg(T* data, int n){
	double sum = 0.0;
	for(int i = 0; i < n; i++){
		sum += data[i];
	}
	return (T) (sum / n);
}

template<typename T>
float calcSTD(T* data, int n, T avg){
	double sum = 0.0;
		for(int i = 0; i < n; i++){
			sum += pow((double)(data[i] - avg), 2);
		}
	return (T) sqrt(sum / n);
}


float montecarloHost1D(int N, double a, double b, double (*f)(double)){
	std::vector<double> xs(N);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(a,b);



	auto gen = std::bind(distribution, generator);
	auto start = std::chrono::steady_clock::now();
	// pontok generálása
	for(auto iter = xs.begin(); iter!= xs.end(); iter++) *iter = gen();

	// le vannak generálva a pontok



	double sum = 0.0f;

	for(double x:xs){
		double y = f(x);
		sum += y;
	}
	double result = (b - a) * sum / N;
	auto end = std::chrono::steady_clock::now();

	auto dt = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

	cout<<"duration: "<<dt.count()<<"us"<<endl;

	return result;
}
void measuremidpointrule1d(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float *res_d;
	float res_h;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));

	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;

	//float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		midpointrule<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Midpointrule iteration: "<<i<<"\t | result: "<<res_h<<"\t |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;
		//avg += et / n_meas;

		samples[i] = res_h;
		ets[i] = et;
	}
	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_samples: "<<avgsamples<<""<<endl;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_samples: "<<sdsamples<<"ms"<<endl;

	CHECK_CUDA_ERROR(cudaFree(res_d));
	cudaDeviceSynchronize();
}
void measuremidpointrule1dFMA(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float *res_d;
	float res_h;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		midpointruleFMA<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"MidpointruleFMA iteration: "<<i<<"\t | result: "<<res_h<<"\t | error: "<<trueval - res_h <<" \t| elapsed time: "<<et<<"ms"<<endl;
	}



	CHECK_CUDA_ERROR(cudaFree(res_d));
	cudaDeviceSynchronize();
}
void measuretrapezoidrule1d(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float *res_d;
	float res_h;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
	float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		trapezoidrule<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Trapezoidrule iteration: "<<i<<"\t | result: "<<res_h<<"\t |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;
		avg += et/n_meas;
	}
	cout<< "AVG: "<<avg<<"ms"<<endl;
	CHECK_CUDA_ERROR(cudaFree(res_d));
	cudaDeviceSynchronize();
}
void measureSimpsonsrule1d(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float *res_d;
	float res_h;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
	float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		simpsonsrule<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Simpsonsrule iteration: "<<i<<"\t | result: "<<res_h<<"\t |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;
		avg += et/n_meas;
	}
	cout<< "AVG: "<<avg<<"ms"<<endl;
	CHECK_CUDA_ERROR(cudaFree(res_d));
	cudaDeviceSynchronize();
}
void measureSimpsonsruleFMA1d(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float *res_d;
	float res_h;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
	float avg;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		simpsonsruleFMA<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"SimpsonsruleFMA iteration: "<<i<<"\t | result: "<<res_h<<"\t |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;
		avg += et/n_meas;
	}
	cout<< "AVG: "<<avg<<"ms"<<endl;
	CHECK_CUDA_ERROR(cudaFree(res_d));
	cudaDeviceSynchronize();
}
void measuremontecarloD1d(int n_meas, int N, double a, double b, float trueval){
	cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandState *state;

	float *res_d;
	float res_h;
	int *offs;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&offs, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandState)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
	int cntr =0;
	float avg = 0.0f;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandState)));
		CHECK_CUDA_ERROR(cudaMemcpy(offs, &cntr, sizeof(int), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		setupmon<<<blocksPerGrid, threadsPerBlock>>>(state, N, offs);
		montecarloD1d<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"iteration: "<<i<<"\t | result: "<<res_h * (b - a) / N<<"\t |error: "<<trueval - (res_h * (b - a) / N) <<" elapsed time: "<<et<<"ms"<<endl;
		CHECK_CUDA_ERROR(cudaFree(state));
		avg += et/n_meas;

		samples[i] = trueval -  (res_h * (b - a) / N);
		ets[i] = et;
		cntr++;
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_samples: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_samples: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(offs));
	cudaDeviceSynchronize();



}
void measuremontecarloDnd(int n_meas, int N, double a, double b, float trueval, int n){
	cudaDeviceReset();
	cudaEvent_t start, stop;



	curandState *state;

	float *res_d;
	int *off;
	float res_h;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));




	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (float *vs){ return sin(vs[0]); };
	float avg = 0.0f;
	int cntr = 0;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));
	for(int i = 0; i < n_meas; i++){



		CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandState)));

		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));


		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		setupmon<<<blocksPerGrid, threadsPerBlock>>>(state, N, off);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDnd<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();


		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;

		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;

		CHECK_CUDA_ERROR(cudaFree(res_d));
		CHECK_CUDA_ERROR(cudaFree(off));
		CHECK_CUDA_ERROR(cudaFree(state));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	//for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_error: "<<sdsamples<<""<<endl;





}
void measuremontecarloDndPhil(int n_meas, int N, double a, double b, float trueval, int n){
	//cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandStatePhilox4_32_10_t *state;

	float *res_d;
	float res_h;
	int *off;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandStatePhilox4_32_10_t)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (float *vs){ return sin(vs[0]); };
	float avg = 0.0f;
	int cntr = 0;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));

		setupmonPhil<<<blocksPerGrid, threadsPerBlock>>>(state, N, off);
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDndPhil<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<"Phil iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;
		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	//for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_error: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(off));
	//cudaDeviceSynchronize();


}
void measuremontecarloDndMrg(int n_meas, int N, double a, double b, float trueval, int n){
	//cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandStateMRG32k3a *state;

	float *res_d;
	float res_h;
	int *off;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandStateMRG32k3a)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (float *vs){ return sin(vs[0]); };
	float avg = 0.0f;
	int cntr  = 0;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));

		setupmonMrg<<<blocksPerGrid, threadsPerBlock>>>(state, N, off);
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDndMrg<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<" Mrg iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;

		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	//for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_error: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(off));
	cudaDeviceSynchronize();


}

void measuremontecarloDndSobol(int n_meas, int N, double a, double b, float trueval, int n){
	//cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandStateSobol32 *state;
	unsigned int *devDirectionVectors;
	curandDirectionVectors32_t *hostVectors;

	curandStatus_t cstatus;
	cstatus = curandGetDirectionVectors32(&hostVectors, CURAND_DIRECTION_VECTORS_32_JOEKUO6);
	if(cstatus == CURAND_STATUS_SUCCESS) cout<< "CURAND_STATUS_SUCCESS"<<endl;
	else if(cstatus == CURAND_STATUS_OUT_OF_RANGE) cout<< "CURAND_STATUS_OUT_OF_RANGE"<<endl;
	else cout<< "CURAND_STATUS_NOT_KNOWN"<<endl;


	float *res_d;
	float res_h;
	int *off;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*n*sizeof(curandStateSobol32)));
	cout<<"Checkpoint"<<endl;
	CHECK_CUDA_ERROR(cudaMalloc((void **)&(devDirectionVectors), n * 32 * sizeof(unsigned int)));
	cout<<"Checkpoint2"<<endl;
	cudaMemcpy(devDirectionVectors, hostVectors, n*32*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cout<<"Checkpoint3"<<endl;

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (float *vs){ return sin(vs[0]); };
	float avg = 0.0f;
	int cntr  = 0;
	cout<<"Checkpoint4"<<endl;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));
		cout<<"Setup_start"<<endl;
		setupmonSobol<<<blocksPerGrid, threadsPerBlock>>>(state, devDirectionVectors, N, off, n);
		cout<<"Setup_done"<<endl;
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDndSobol<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<" Sobol iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;

		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	//for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	float sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_error: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(off));
	cudaDeviceSynchronize();


}

void measuremontecarloDndMt(int n_meas, int N, double a, double b, float trueval, int n){
	//cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandStateMtgp32_t *state;
	mtgp32_kernel_params *devKernelParams;

	float *res_d;
	float res_h;
	int *off;

	float *samples, *ets;
	samples = (float*)malloc(N*sizeof(float));
	ets = (float*)malloc(N*sizeof(float));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandStateMtgp32_t)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));
	curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);

	CHECK_LAST_CUDA_ERROR();
	curandMakeMTGP32KernelState(state, mtgp32dc_params_fast_11213, devKernelParams, 64, 1234);
	CHECK_LAST_CUDA_ERROR();
	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (float *vs){ return sin(vs[0]); };
	float avg = 0.0f;
	int cntr  = 0;
	for(int i = 0; i < n_meas; i++){
		CHECK_LAST_CUDA_ERROR();
		cout<<"Start"<<endl;
		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDndMt<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
		cout<<" Mrg iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;

		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;
	}

	float avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	float sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	float avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	float sdsamples = calcSTD(samples, n_meas, avgsamples / trueval);
	cout<< "SD_error: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(devKernelParams));
	CHECK_CUDA_ERROR(cudaFree(off));
	cudaDeviceSynchronize();


}

void measuremontecarloDndMrgDBL(int n_meas, int N, double a, double b, double trueval, int n){
	//cudaDeviceReset();
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	curandStateMRG32k3a *state;

	double *res_d;
	double res_h;
	int *off;

	double *samples, *ets;
	samples = (double*)malloc(N*sizeof(double));
	ets = (double*)malloc(N*sizeof(double));


	CHECK_CUDA_ERROR(cudaMalloc((void**)&res_d, sizeof(double)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&off, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&state, N*sizeof(curandStateMRG32k3a)));

	const int BLOCK_SIZE = 256;

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocksPerGrid(N / BLOCK_SIZE + 1);
	float et;
	cudaError_t e;
    //auto o = [] __device__ (double *vs){ return sin(vs[0]); };
	double avg = 0.0f;
	int cntr  = 0;
	for(int i = 0; i < n_meas; i++){
		CHECK_CUDA_ERROR(cudaMemcpy(off, &cntr, sizeof(int), cudaMemcpyHostToDevice));

		setupmonMrg<<<blocksPerGrid, threadsPerBlock>>>(state, N, off);
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		montecarloDndMrgDBL<<<blocksPerGrid, threadsPerBlock>>>(N, res_d, state, n, a, b);
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();

		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&et, start, stop));

		CHECK_CUDA_ERROR(cudaMemcpy(&res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost));
		cout<<" Mrg iteration: "<<i<<"\t | result: "<<res_h<<"\t | |error: "<<trueval - res_h <<" elapsed time: "<<et<<"ms"<<endl;

		samples[i] = trueval - res_h;
		ets[i] = et;
		cntr++;
	}

	double avget = calcavg(ets, n_meas);
	cout<< "AVG_et: "<<avget<<"ms"<<endl;
	double sdet = calcSTD(ets, n_meas, avget);
	cout<< "SD_et: "<<sdet<<"ms"<<endl;

	double avgsamples = calcavg(samples, n_meas);
	cout<< "AVG_error: "<<avgsamples<<""<<endl;
	cout<< "AVG_rel_error: "<<avgsamples / trueval<<""<<endl;
	//for(int i = 0;i<n_meas;i++)samples[i] /=trueval;
	double sdsamples = calcSTD(samples, n_meas, avgsamples);
	cout<< "SD_error: "<<sdsamples<<""<<endl;


	CHECK_CUDA_ERROR(cudaFree(res_d));
	CHECK_CUDA_ERROR(cudaFree(off));
	cudaDeviceSynchronize();


}


int main() {
	double a, b;
	a = 0.0;
	b = M_PI;
	double trueval = 0.0;
	int n_b = 1000 ;
	cout<<"------------Host Mont D1d-----------"<<endl;
	double res = montecarloHost1D(n_b, a, b , sin);
	std::cout << std::setprecision(32);
	cout<< "result: " <<res<<endl;


	cout<<"------------Device midpointrule1d-----------"<<endl;
	measuremidpointrule1d(10, n_b, a, b, trueval);
	cout<<"------------Device midpointrule1dFMA-----------"<<endl;
	measuremidpointrule1dFMA(10, n_b, a, b, trueval);
	cout<<"------------Device trapezoidrule-----------"<<endl;
	measuretrapezoidrule1d(10, n_b, a, b, trueval);
	cout<<"------------Device simpsonsrule-----------"<<endl;
	measureSimpsonsrule1d(10, n_b, a, b, trueval);
	cout<<"------------Device simpsonsruleFMA-----------"<<endl;
	measureSimpsonsruleFMA1d(10, n_b, a, b, trueval);


	//cout<<"------------Device D1d-----------"<<endl;
	//measuremontecarloD1d(100, 10, a, b, trueval);

	//trueval = 2 * (b-a);
	trueval = 0.0;
	a = 0;
	//b = M_PI;
	b = 2*sqrt(M_PI);
	int n_points = 1000000;
	int ndim = 2;
	cout<<"Trueval: "<<trueval<<endl;
	cout<<"------------Device Dnd-----------"<<endl;
	measuremontecarloDnd(10, n_points, a, b, trueval, ndim);
	cout<<endl;

	cout<<"------------Device DndPhil-----------"<<endl;
	measuremontecarloDndPhil(10, n_points, a, b,  trueval, ndim);
	cout<<endl;

	cout<<"------------Device DndMrg-----------"<<endl;
	measuremontecarloDndMrg(10, n_points, a, b,  trueval, ndim);
	cout<<endl;
	//cout<<"------------Device DndMrgDBL-----------"<<endl;
	//measuremontecarloDndMrgDBL(10, n_points, a, b,  trueval, ndim);

	//cout<<"------------Device DndMt-----------"<<endl;
	//measuremontecarloDndMt(1, n_points, a, b,  trueval, ndim);

	cout<<"------------Device DndSobol-----------"<<endl;
	measuremontecarloDndSobol(10, n_points, a, b,  trueval, ndim);
	cout<<endl;


	return 0;
}
