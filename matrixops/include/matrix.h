#pragma once
#include "cuda_runtime.h"
#include <cuda.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;




void VecAdd_h(float* A, float* B, float* C, int N);
void ArraySum_h(float* v, float* result, int N);
__global__ void VecAdd(float* A, float* B, float* C, int N);
__global__ void ArraySum(float* v, float* result, int N);
__global__ void ArraySumBin(float* v, float* result, int N);


__global__ void MatAdd(Matrix A, Matrix B, Matrix C);
__global__ void MatMulSimple(Matrix A, Matrix B, Matrix C);
__global__ void MatMulSimpleFMA(Matrix A, Matrix B, Matrix C);
void MatMulHost(Matrix A, Matrix B, Matrix C);

void printVec(float* v, int N);
void initVec(float* v, float val, int N);

void MatAdd_h(Matrix A, Matrix B, Matrix C);

void printMat(Matrix A, int N1, int N2);
void initMat(Matrix* A, float val);

void initMat_d(Matrix* A);
