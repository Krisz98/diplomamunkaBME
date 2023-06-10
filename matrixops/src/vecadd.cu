// matrixoperations.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "cuda_runtime.h"
#include <cuda.h>
#include <chrono>
#include<assert.h>

#include "matrix.h"

#include <fstream>

constexpr float EPSILON = 0.1f;


std::chrono::nanoseconds measureVecAdd(float* A, float* B, float* C, int N, int threadsPerBlock, int blocksPerGrid) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    VecAdd << <blocksPerGrid, threadsPerBlock >> > (A, B, C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}
std::chrono::nanoseconds measureArrsum(float* A, float* result, int N, int threadsPerBlock, int blocksPerGrid) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    ArraySum << <blocksPerGrid, threadsPerBlock >> > (A,result,N);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}
std::chrono::nanoseconds measureArrsumBin(float* A, float* result, int N, int threadsPerBlock, int blocksPerGrid) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    ArraySumBin << <blocksPerGrid, threadsPerBlock >> > (A, result, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}
std::chrono::nanoseconds measureArrsum_h(float* A, float* result, int N) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    ArraySum_h(A, result, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

std::chrono::nanoseconds measureMatAdd(Matrix A, Matrix B, Matrix C, dim3 gridSize, dim3 matBlockSize) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    MatAdd << <gridSize, matBlockSize >> > (A, B, C);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}
std::chrono::nanoseconds measureMatMulSimple(Matrix A, Matrix B, Matrix C, dim3 gridSize, dim3 matBlockSize) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    MatMulSimple << <gridSize, matBlockSize >> > (A, B, C);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

std::chrono::nanoseconds measureMatMulHost(Matrix A, Matrix B, Matrix C) {
    auto start = std::chrono::steady_clock::now();
    MatMulHost(A, B, C);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

void testSimpleMul(Matrix A, Matrix B, Matrix C, dim3 gridSize, dim3 matBlockSize){
	cudaDeviceSynchronize();
	MatMulSimple << <gridSize, matBlockSize >> > (A, B, C);
	cudaDeviceSynchronize();
}

void testSImpleMulFMA(Matrix A, Matrix B, Matrix C, dim3 gridSize, dim3 matBlockSize){
	cudaDeviceSynchronize();
	MatMulSimple << <gridSize, matBlockSize >> > (A, B, C);
	cudaDeviceSynchronize();
}





void checkResult(float e, float* r, int N) {
    for (int i = 0;i < N;i++) {
        assert(r[i] == e);
    }
}

int main()
{
    
    std::ofstream outputfile("results.csv");
    outputfile << "Iteration,N,copytodevice,calcd1,arraysumbin,arraysumh,arraysumt,copytohost,calch" << std::endl;

    const int MAXITER = log10(10000000);
    const int iters[] = { 10,2010,4010,6010,8010,10010,12010,14010,16010,18010,20010,22010,24010,26010,28010,30010,32010,34010,36010,38010,40010,42010,44010,46010,48010,50010,52010,54010,56010,58010,60010,62010,64010,66010,68010,70010,72010,74010,76010,78010,80010,82010,84010,86010,88010,90010,92010,94010,96010,98010,1.0001e+05,1.0201e+05,1.0401e+05,1.0601e+05,1.0801e+05,1.1001e+05,1.1201e+05,1.1401e+05,1.1601e+05,1.1801e+05,1.2001e+05,1.2201e+05,1.2401e+05,1.2601e+05,1.2801e+05,1.3001e+05,1.3201e+05,1.3401e+05,1.3601e+05,1.3801e+05,1.4001e+05,1.4201e+05,1.4401e+05,1.4601e+05,1.4801e+05,1.5001e+05,1.5201e+05,1.5401e+05,1.5601e+05,1.5801e+05,1.6001e+05,1.6201e+05,1.6401e+05,1.6601e+05,1.6801e+05,1.7001e+05,1.7201e+05,1.7401e+05,1.7601e+05,1.7801e+05,1.8001e+05,1.8201e+05,1.8401e+05,1.8601e+05,1.8801e+05,1.9001e+05,1.9201e+05,1.9401e+05,1.9601e+05,1.9801e+05,2.0001e+05,2.0201e+05,2.0401e+05,2.0601e+05,2.0801e+05,2.1001e+05,2.1201e+05,2.1401e+05,2.1601e+05,2.1801e+05,2.2001e+05,2.2201e+05,2.2401e+05,2.2601e+05,2.2801e+05,2.3001e+05,2.3201e+05,2.3401e+05,2.3601e+05,2.3801e+05,2.4001e+05,2.4201e+05,2.4401e+05,2.4601e+05,2.4801e+05,2.5001e+05,2.5201e+05,2.5401e+05,2.5601e+05,2.5801e+05,2.6001e+05,2.6201e+05,2.6401e+05,2.6601e+05,2.6801e+05,2.7001e+05,2.7201e+05,2.7401e+05,2.7601e+05,2.7801e+05,2.8001e+05,2.8201e+05,2.8401e+05,2.8601e+05,2.8801e+05,2.9001e+05,2.9201e+05,2.9401e+05,2.9601e+05,2.9801e+05,3.0001e+05,3.0201e+05,3.0401e+05,3.0601e+05,3.0801e+05,3.1001e+05,3.1201e+05,3.1401e+05,3.1601e+05,3.1801e+05,3.2001e+05,3.2201e+05,3.2401e+05,3.2601e+05,3.2801e+05,3.3001e+05,3.3201e+05,3.3401e+05,3.3601e+05,3.3801e+05,3.4001e+05,3.4201e+05,3.4401e+05,3.4601e+05,3.4801e+05,3.5001e+05,3.5201e+05,3.5401e+05,3.5601e+05,3.5801e+05,3.6001e+05,3.6201e+05,3.6401e+05,3.6601e+05,3.6801e+05,3.7001e+05,3.7201e+05,3.7401e+05,3.7601e+05,3.7801e+05,3.8001e+05,3.8201e+05,3.8401e+05,3.8601e+05,3.8801e+05,3.9001e+05,3.9201e+05,3.9401e+05,3.9601e+05,3.9801e+05,4.0001e+05,4.0201e+05,4.0401e+05,4.0601e+05,4.0801e+05,4.1001e+05,4.1201e+05,4.1401e+05,4.1601e+05,4.1801e+05,4.2001e+05,4.2201e+05,4.2401e+05,4.2601e+05,4.2801e+05,4.3001e+05,4.3201e+05,4.3401e+05,4.3601e+05,4.3801e+05,4.4001e+05,4.4201e+05,4.4401e+05,4.4601e+05,4.4801e+05,4.5001e+05,4.5201e+05,4.5401e+05,4.5601e+05,4.5801e+05,4.6001e+05,4.6201e+05,4.6401e+05,4.6601e+05,4.6801e+05,4.7001e+05,4.7201e+05,4.7401e+05,4.7601e+05,4.7801e+05,4.8001e+05,4.8201e+05,4.8401e+05,4.8601e+05,4.8801e+05,4.9001e+05,4.9201e+05,4.9401e+05,4.9601e+05,4.9801e+05,5.0001e+05,5.0201e+05,5.0401e+05,5.0601e+05,5.0801e+05,5.1001e+05,5.1201e+05,5.1401e+05,5.1601e+05,5.1801e+05,5.2001e+05,5.2201e+05,5.2401e+05,5.2601e+05,5.2801e+05,5.3001e+05,5.3201e+05,5.3401e+05,5.3601e+05,5.3801e+05,5.4001e+05,5.4201e+05,5.4401e+05,5.4601e+05,5.4801e+05,5.5001e+05,5.5201e+05,5.5401e+05,5.5601e+05,5.5801e+05,5.6001e+05,5.6201e+05,5.6401e+05,5.6601e+05,5.6801e+05,5.7001e+05,5.7201e+05,5.7401e+05,5.7601e+05,5.7801e+05,5.8001e+05,5.8201e+05,5.8401e+05,5.8601e+05,5.8801e+05,5.9001e+05,5.9201e+05,5.9401e+05,5.9601e+05,5.9801e+05,6.0001e+05,6.0201e+05,6.0401e+05,6.0601e+05,6.0801e+05,6.1001e+05,6.1201e+05,6.1401e+05,6.1601e+05,6.1801e+05,6.2001e+05,6.2201e+05,6.2401e+05,6.2601e+05,6.2801e+05,6.3001e+05,6.3201e+05,6.3401e+05,6.3601e+05,6.3801e+05,6.4001e+05,6.4201e+05,6.4401e+05,6.4601e+05,6.4801e+05,6.5001e+05,6.5201e+05,6.5401e+05,6.5601e+05,6.5801e+05,6.6001e+05,6.6201e+05,6.6401e+05,6.6601e+05,6.6801e+05,6.7001e+05,6.7201e+05,6.7401e+05,6.7601e+05,6.7801e+05,6.8001e+05,6.8201e+05,6.8401e+05,6.8601e+05,6.8801e+05,6.9001e+05,6.9201e+05,6.9401e+05,6.9601e+05,6.9801e+05,7.0001e+05,7.0201e+05,7.0401e+05,7.0601e+05,7.0801e+05,7.1001e+05,7.1201e+05,7.1401e+05,7.1601e+05,7.1801e+05,7.2001e+05,7.2201e+05,7.2401e+05,7.2601e+05,7.2801e+05,7.3001e+05,7.3201e+05,7.3401e+05,7.3601e+05,7.3801e+05,7.4001e+05,7.4201e+05,7.4401e+05,7.4601e+05,7.4801e+05,7.5001e+05,7.5201e+05,7.5401e+05,7.5601e+05,7.5801e+05,7.6001e+05,7.6201e+05,7.6401e+05,7.6601e+05,7.6801e+05,7.7001e+05,7.7201e+05,7.7401e+05,7.7601e+05,7.7801e+05,7.8001e+05,7.8201e+05,7.8401e+05,7.8601e+05,7.8801e+05,7.9001e+05,7.9201e+05,7.9401e+05,7.9601e+05,7.9801e+05,8.0001e+05,8.0201e+05,8.0401e+05,8.0601e+05,8.0801e+05,8.1001e+05,8.1201e+05,8.1401e+05,8.1601e+05,8.1801e+05,8.2001e+05,8.2201e+05,8.2401e+05,8.2601e+05,8.2801e+05,8.3001e+05,8.3201e+05,8.3401e+05,8.3601e+05,8.3801e+05,8.4001e+05,8.4201e+05,8.4401e+05,8.4601e+05,8.4801e+05,8.5001e+05,8.5201e+05,8.5401e+05,8.5601e+05,8.5801e+05,8.6001e+05,8.6201e+05,8.6401e+05,8.6601e+05,8.6801e+05,8.7001e+05,8.7201e+05,8.7401e+05,8.7601e+05,8.7801e+05,8.8001e+05,8.8201e+05,8.8401e+05,8.8601e+05,8.8801e+05,8.9001e+05,8.9201e+05,8.9401e+05,8.9601e+05,8.9801e+05,9.0001e+05,9.0201e+05,9.0401e+05,9.0601e+05,9.0801e+05,9.1001e+05,9.1201e+05,9.1401e+05,9.1601e+05,9.1801e+05,9.2001e+05,9.2201e+05,9.2401e+05,9.2601e+05,9.2801e+05,9.3001e+05,9.3201e+05,9.3401e+05,9.3601e+05,9.3801e+05,9.4001e+05,9.4201e+05,9.4401e+05,9.4601e+05,9.4801e+05,9.5001e+05,9.5201e+05,9.5401e+05,9.5601e+05,9.5801e+05,9.6001e+05,9.6201e+05,9.6401e+05,9.6601e+05,9.6801e+05,9.7001e+05,9.7201e+05,9.7401e+05,9.7601e+05,9.7801e+05,9.8001e+05,9.8201e+05,9.8401e+05,9.8601e+05,9.8801e+05,9.9001e+05,9.9201e+05,9.9401e+05,9.9601e+05,9.9801e+05,1e+06,1.5e+06,2e+06,2.5e+06,3e+06,3.5e+06,4e+06,4.5e+06,5e+06,5.5e+06,6e+06,1e+06,6e+06,1.1e+07,1.6e+07,2.1e+07,2.6e+07,3.1e+07,3.6e+07,4.1e+07,4.6e+07,5.1e+07,5.6e+07,6.1e+07,6.6e+07,7.1e+07,7.6e+07,8.1e+07,8.6e+07,9.1e+07,9.6e+07,
        268000000
    };
    std::cout << "Number of iterations: " << sizeof(iters)/sizeof(int) << std::endl;
    for (int i = 0; i < sizeof(iters)/sizeof(int); i++) {
        
        //resetelem a context-et
        cudaDeviceReset();  
        //inicializ�lom a v�ltoz�kat a host-on
        int N = iters[i];
        size_t size = N * sizeof(float);

        float* h_A = (float*)malloc(size);
        float* h_B = (float*)malloc(size);
        float* h_C = (float*)malloc(size);
        float* h_vec = (float*)malloc(size);
        float res;
        float res_h;
        float res_bin_h;
        
        float initVal1 = 1.2;  //be�ll�tom mire fogom inicializ�lni a vektorokat
        float initVal2 = 2.0;
        float initValvec = 0.01;
        initVec(h_A, initVal1, N);
        initVec(h_B, initVal2, N);
        initVec(h_vec, initValvec, N);

        // foglalok mem�ri�t a GPU-n
        float* d_A; cudaMalloc(&d_A, size);
        float* d_B; cudaMalloc(&d_B, size);
        float* d_C; cudaMalloc(&d_C, size);
        float* d_vec; cudaMalloc(&d_vec, size);
        float* d_res; cudaMalloc(&d_res, sizeof(float));
        float* d_res_bin; cudaMalloc(&d_res_bin, sizeof(float));


        cudaMemcpy(d_vec,h_vec,size,cudaMemcpyHostToDevice);

        // �tm�solom a t�mb�k tartalm�t a GPU-ra
        auto start = std::chrono::steady_clock::now();
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        auto end = std::chrono::steady_clock::now();
        std::chrono::nanoseconds memcpyTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // elind�tom a kernel-t
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;


        std::chrono::nanoseconds vt1 = measureVecAdd(d_A, d_B, d_C, N, threadsPerBlock, blocksPerGrid);
        std::chrono::nanoseconds arraySumt_h = measureArrsum_h(h_vec,&res_h,N);
        std::chrono::nanoseconds arraysumt = measureArrsum(d_vec,d_res,N,threadsPerBlock,blocksPerGrid);
        std::chrono::nanoseconds arraySumBint = measureArrsumBin(d_vec, d_res_bin, N, threadsPerBlock, blocksPerGrid);

        start = std::chrono::steady_clock::now();
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        end = std::chrono::steady_clock::now();
        std::chrono::nanoseconds cpyback = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        start = std::chrono::steady_clock::now();
        VecAdd_h(h_A, h_B, h_C, N);
        end = std::chrono::steady_clock::now();
        std::chrono::nanoseconds hostVecAddTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&res_bin_h, d_res_bin, sizeof(float), cudaMemcpyDeviceToHost);

        

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_vec);
        cudaFree(d_res);
        cudaFree(d_res_bin);

        outputfile << i << "," << N << "," << memcpyTime.count() << "," << vt1.count() << "," << arraySumBint.count() << "," << arraySumt_h.count() << "," << arraysumt.count() << "," <<
            cpyback.count() << "," << hostVecAddTime.count() << std::endl;

        std::cout << std::endl<<"Iteration: " << i << std::endl;
        std::cout << "N = " << N << std::endl;
        std::cout << "ArrayDesiredSum: " << N*initValvec << std::endl;
        std::cout << "ArraySum: " << res << std::endl;
        std::cout << "ArraySum_h: " << res_h << std::endl;
        std::cout << "ArraySumBin: " << res_bin_h << std::endl;
        std::cout << "mem copy time: " << memcpyTime.count() << std::endl;
        std::cout << "cuda vecAdd time: " << vt1.count() << std::endl;
        std::cout << "cuda arraySum_h time: " << arraySumt_h.count() << std::endl;
        std::cout << "cuda arraySum time: " << arraysumt.count() << std::endl;
        std::cout << "cuda ArraySumBin time: " << arraySumBint.count() << std::endl;
        std::cout << "host vecAdd time: " << hostVecAddTime.count() << std::endl;
        //assert((res < (N * initValvec + EPSILON)) && (res > (N * initValvec - EPSILON)));

        checkResult(initVal1 + initVal2, h_C, N);

        free(h_A);
        free(h_B);
        free(h_C);
        free(h_vec);


    }

    outputfile.close();









    Matrix A, B, C;

    A.width = 1000;
    A.height = 2000;

    B.width = 3000;
    B.height = 1000;

    C.width = 3000;
    C.height = 2000;

    initMat(&A, 1.0);
    initMat(&B, 0.002);
    initMat(&C, 0.0);
    
    printMat(A,3,4);

    Matrix dmA, dmB, dmC;

    dmA.width =A.width;
    dmA.height = A.height;
    dmB.width = B.width;
    dmB.height = B.height;
    dmC.width = C.width;
    dmC.height = C.height;

    initMat_d(&dmA);
    initMat_d(&dmB);
    initMat_d(&dmC);

    cudaMemcpy((dmA.elements), (A.elements), A.width  * A.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((dmB.elements), (B.elements), B.width * B.height * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 matBlockSize(4,5);
    dim3 gridSize(C.height / matBlockSize.x, C.width / matBlockSize.y);

    std::chrono::nanoseconds mt1 = measureMatMulSimple(dmA,dmB,dmC, gridSize,matBlockSize);
    std::chrono::nanoseconds mt2 = measureMatMulSimple(dmA, dmB, dmC, gridSize, matBlockSize);
    std::chrono::nanoseconds mt3 = measureMatMulSimple(dmA, dmB, dmC, gridSize, matBlockSize);
    std::chrono::nanoseconds mt4 = measureMatMulSimple(dmA, dmB, dmC, gridSize, matBlockSize);
    cudaMemcpy((C.elements), (dmC.elements), C.width * C.height * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "result:" << std::endl;
    printMat(C, 3,4);

    initMat(&C, 0.0);
    //auto start = std::chrono::steady_clock::now();
    std::chrono::nanoseconds hostMatAddTime = measureMatMulHost(A, B, C);
    //auto end = std::chrono::steady_clock::now();
    //std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Result:" << std::endl;
    printMat(C,3,4);

    std::cout << "cuda matAdd time: " << mt1.count() << std::endl;
    std::cout << "cuda matAdd time: " << mt2.count() << std::endl;
    std::cout << "cuda matAdd time: " << mt3.count() << std::endl;
    std::cout << "cuda matAdd time: " << mt4.count() << std::endl;
    std::cout << "host matAdd time: " << hostMatAddTime.count() << std::endl;

    free(A.elements);
    free(B.elements);
    free(C.elements);

    cudaFree(dmA.elements);
    cudaFree(dmB.elements);
    cudaFree(dmC.elements);



    

    return 0;

    
}

