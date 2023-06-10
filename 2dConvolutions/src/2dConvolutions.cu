//============================================================================
// Name        : 2dConvolutions.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include "convs2d.h"
using namespace cv;

#define N_MEASURE 10


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

std::vector<uchar> to_array_v1(cv::Mat const& img)
{
    std::vector<uchar> a;
    if (img.isContinuous()) {
        img.reshape(1, 1).copyTo(a);
    }
    return a;
}

void processimage(Mat &input, Mat &output){
	std::vector<uchar> vimg;
	vimg = to_array_v1(input);
	unsigned char *imgp = &vimg[0];
	unsigned char imgres[input.rows*input.cols];

	float kernel[KERNEL_WIDTH*KERNEL_HEIGHT] = BOXBLUR_KERNEL;

	int size_img = input.rows * input.cols;

	int asize = input.rows * input.cols;
	unsigned char *d_img, *d_img_res;
	CHECK_CUDA_ERROR(cudaMalloc(&d_img, asize*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_img_res, asize*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel, kernel, KERNEL_HEIGHT*KERNEL_WIDTH*sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_img, imgp, size_img*sizeof(unsigned char), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(input.cols/BLOCK_SIZE + 1, input.rows/BLOCK_SIZE + 1);

	Conv2d::c2dS<<<numBlocks, threadsPerBlock>>>(d_img, d_img_res, input.cols, input.rows);

	CHECK_CUDA_ERROR(cudaMemcpy(imgres, d_img_res, size_img*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaFree(d_img));
	CHECK_CUDA_ERROR(cudaFree(d_img_res));

	CHECK_LAST_CUDA_ERROR();
	CHECK_LAST_CUDA_ERROR();

	// reconstruct image
	Mat res_img(input.rows, input.cols, CV_8UC1, (unsigned char*)imgres);
	res_img.copyTo(output);

}
void processimage2step(Mat &input, Mat &output){
	std::vector<uchar> vimg;
	vimg = to_array_v1(input);
	unsigned char *imgp = &vimg[0];
	unsigned char imgres[input.rows*input.cols];

	float kernel_horizontal[KERNEL_WIDTH] = HORIZONTAL_IDENTITY_KERNEL;
	float kernel_vertical[KERNEL_HEIGHT] = VERTICAL_IDENTITY_KERNEL;

	int size_img = input.rows * input.cols;

	int asize = input.rows * input.cols;
	unsigned char *d_img, *d_img_res;
	CHECK_CUDA_ERROR(cudaMalloc(&d_img, asize*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_img_res, asize*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel_horizontal, kernel_horizontal, KERNEL_WIDTH*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel_vertical, kernel_vertical, KERNEL_HEIGHT*sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_img, imgp, size_img*sizeof(unsigned char), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(input.cols/BLOCK_SIZE, input.rows/BLOCK_SIZE);
	Conv2d::c2dSHorizontal<<<numBlocks, threadsPerBlock>>>(d_img, d_img_res, input.cols, input.rows);
	Conv2d::c2dSVertical<<<numBlocks, threadsPerBlock>>>(d_img_res, d_img_res, input.cols, input.rows);
	CHECK_CUDA_ERROR(cudaMemcpy(imgres, d_img_res, size_img*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaFree(d_img));
	CHECK_CUDA_ERROR(cudaFree(d_img_res));

	CHECK_LAST_CUDA_ERROR();
	CHECK_LAST_CUDA_ERROR();

	// reconstruct image
	Mat res_img(input.rows, input.cols, CV_8UC1, (unsigned char*)imgres);
	res_img.copyTo(output);

}



void measureC2dG(unsigned char *input, int width, int height, float &avg){
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float kernel[KERNEL_WIDTH*KERNEL_HEIGHT] = SHARPEN_KERNEL;
	unsigned char *d_img, *d_img_res;

	CHECK_CUDA_ERROR(cudaMalloc(&d_img, width*height*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_img_res, width*height*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel, kernel, KERNEL_HEIGHT*KERNEL_WIDTH*sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_img, input, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(width/BLOCK_SIZE, height/BLOCK_SIZE);
	avg = 0.0f;
	for(int i = 0; i < N_MEASURE; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));

		Conv2d::c2dS2<<<numBlocks, threadsPerBlock>>>(d_img, d_img_res, width, height);

		//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout<<"elapsed time: "<<milliseconds<<" ms"<<std::endl;
		avg += milliseconds / N_MEASURE;
	}

	CHECK_CUDA_ERROR(cudaFree(d_img));
	CHECK_CUDA_ERROR(cudaFree(d_img_res));

	CHECK_LAST_CUDA_ERROR();
	CHECK_LAST_CUDA_ERROR();

}

void measureC2dSSeparable(unsigned char *input, int width, int height, float &avg){
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaDeviceReset());
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));


	float kernel_horizontal[KERNEL_WIDTH] = HORIZONTAL_IDENTITY_KERNEL;
	float kernel_vertical[KERNEL_HEIGHT] = VERTICAL_IDENTITY_KERNEL;
	unsigned char *d_img;
	unsigned char *d_img_res;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_img, width*height*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_img_res, width*height*sizeof(unsigned char)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel_horizontal, kernel_horizontal, KERNEL_WIDTH*sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Conv2d::kernel_vertical, kernel_vertical, KERNEL_HEIGHT*sizeof(float)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_img, input, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(width/BLOCK_SIZE, height/BLOCK_SIZE);
	avg = 0.0f;
	for(int i = 0; i < N_MEASURE; i++){
		CHECK_CUDA_ERROR(cudaEventRecord(start));
		Conv2d::c2dSHorizontal_fused<<<numBlocks, threadsPerBlock>>>(d_img, d_img_res, width, height);
		Conv2d::c2dSVertical_fused<<<numBlocks, threadsPerBlock>>>(d_img_res, d_img_res, width, height);
		//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaEventRecord(stop));
		CHECK_LAST_CUDA_ERROR();
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout<<"elapsed time: "<<milliseconds<<" ms"<<std::endl;
		avg += milliseconds / N_MEASURE;
	}

	CHECK_CUDA_ERROR(cudaFree(d_img));
	CHECK_CUDA_ERROR(cudaFree(d_img_res));

	CHECK_LAST_CUDA_ERROR();
	CHECK_LAST_CUDA_ERROR();

}

void processimageHost(Mat &input, Mat &output){
	std::cout<<"type:"<<input.type();
	int t = input.type();
	int kernel_size = 3; //+ 2*( 2%5 );
	Point anchor = Point( -1, -1 );
	Mat kernel = Mat::ones( kernel_size, kernel_size, CV_8U );/// (float)(kernel_size*kernel_size);
	filter2D(input, output, -1 , kernel, anchor, 0, BORDER_DEFAULT );
}

int main( int argc, char** argv )
{
  Mat image;
  image = imread( argv[1], IMREAD_GRAYSCALE );
  if( argc != 2 || !image.data )
    {
      printf( "No image data \n" );
      return -1;
    }

  //std::cout<<"rows: "<< image.rows<<" | cols: "<<image.cols<<std::endl;


  Mat result;
  processimage(image, result);
  //Mat resized;

  //processimageHost(image, result);
  imwrite("/home/krisztian/Documents/cuda/diploma/ferrarioriginal.jpg", image);
  imwrite("/home/krisztian/Documents/cuda/diploma/ferrarisharpened.jpg", result);
  namedWindow( "RESULT", WINDOW_AUTOSIZE );
  imshow( "RESULT", result );
  namedWindow( "ORIGINAL", WINDOW_AUTOSIZE );
  imshow( "ORIGINAL", image );
  waitKey(0);

  unsigned char *inputimg;

  int wh = 19200;
  inputimg = new unsigned char[wh*wh];
  std::memset(inputimg, 50, wh*wh*sizeof(char));
  float avg = 0.0f;
  measureC2dSSeparable(inputimg, wh, wh, avg);
  std::cout<<"avg: "<<avg<<" ms"<<std::endl;
  delete inputimg;

  std::cout<<"Finished";
  return 0;
}
