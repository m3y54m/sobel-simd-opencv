//First we set an enviroment variable to make easier our work.This will hold the build directory of our OpenCV library that we use in our projects.Start up a command window and enter :
//
//setx - m OPENCV_DIR D : \OpenCV\Build\x86\vc10(suggested for Visual Studio 2010 - 32 bit Windows)
//setx - m OPENCV_DIR D : \OpenCV\Build\x64\vc10(suggested for Visual Studio 2010 - 64 bit Windows)
//
//setx - m OPENCV_DIR D : \OpenCV\Build\x86\vc11(suggested for Visual Studio 2012 - 32 bit Windows)
//setx - m OPENCV_DIR D : \OpenCV\Build\x64\vc11(suggested for Visual Studio 2012 - 64 bit Windows)
//
//And add %OPENCV_DIR%\bin to Path in enviroment variables

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <intrin.h>
#include <Windows.h>
//#include <chrono>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	/* C coding style
	//IplImage *imgInput, *imgOutputNonSIMD;

	//imgInput = cvLoadImage("boat.png", CV_LOAD_IMAGE_GRAYSCALE);

	//int width = imgInput->width;
	//int height = imgInput->height;

	//imgOutputNonSIMD = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

	//cvNamedWindow("in", CV_WINDOW_AUTOSIZE);
	//cvShowImage("in", imgInput);
	*/

	// precise time measurement
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTime;
	QueryPerformanceFrequency(&frequency); // get ticks per second
	double microseconds_per_count = 1.0e6 / static_cast<double>(frequency.QuadPart);

	// OpenCV image datatypes
	Mat imgInput, imgOutputNonSIMD, imgOutputSIMD;

	const int border = 8;

	if (border < 1) {
		cout << "border must be greater than or equal to 1" << endl;
		cout << "Press any key to exit!" << endl;
		getchar();
		return 0;
	}

	// load input image
	imgInput = imread("monarch.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	int initialWidth = imgInput.cols;
	int initialHeight = imgInput.rows;

	// show input image
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imgInput);
	
	// add border to image using replication method
	copyMakeBorder(imgInput, imgInput, border, border, border, border, BORDER_REPLICATE);

	/*********** Non-SIMD **********/

	// allocate new array data for imgOutputNonSIMD with the same size as imgInput.
	imgOutputNonSIMD.create(imgInput.size(), imgInput.depth());

	uchar m1, m2, m3, m4, m5, m6, m7, m8, m9;
	int Gx, Gy;

	uchar* outputPointer = imgOutputNonSIMD.ptr<uchar>();
	uchar* inputPointer = imgInput.ptr<uchar>();

	int width = imgInput.cols;
	int height = imgInput.rows;

	// get the current cpu time
	QueryPerformanceCounter(&t1); // start timer
	//auto begin = chrono::high_resolution_clock::now();

	for (int i = (border - 1); i < height - (border + 1); i++) //rows
	{
		for (int j = (border - 1); j < width - (border + 1); j++) // cols
		{

			/*
			Sobel operator input matrix
			+~~~~~~~~~~~~~~+
			| m1 | m2 | m3 |
			|~~~~+~~~~+~~~~+
			| m4 | m5 | m6 |
			|~~~~+~~~~+~~~~+
			| m7 | m8 | m9 |  
			+~~~~+~~~~+~~~~+
			*/

			m1 = *(inputPointer + i * width + j);
			m2 = *(inputPointer + i * width + j + 1);
			m3 = *(inputPointer + i * width + j + 2);

			m4 = *(inputPointer + (i + 1) * width + j);
			m5 = *(inputPointer + (i + 1) * width + j + 1);
			m6 = *(inputPointer + (i + 1) * width + j + 2);

			m7 = *(inputPointer + (i + 2) * width + j);
			m8 = *(inputPointer + (i + 2) * width + j + 1);
			m9 = *(inputPointer + (i + 2) * width + j + 2);

			// Calculating Gx
			Gx = (m3 + 2 * m6 + m9) - (m1 + 2 * m4 + m7);

			// Calculating Gy
			Gy = (m1 + 2 * m2 + m3) - (m7 + 2 * m8 + m9);

			outputPointer[(i + 1) * width + j + 1] = saturate_cast<uchar>(abs(Gx) + abs(Gy)); // approximate
		}
	}

	// get the current cpu time
	QueryPerformanceCounter(&t2); // stop timer
	// calculate and print elapsed time in microseconds
	elapsedTime = static_cast<double>(t2.QuadPart - t1.QuadPart) * microseconds_per_count;
	cout << "Execution time for non-SIMD Sobel edge detection:" << endl;
	cout << elapsedTime << " us" << endl;
	//auto end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "ns" << std::endl;

	// crop image and remove added borders
	Rect cropped(border, border, initialWidth, initialHeight);
	imgOutputNonSIMD = imgOutputNonSIMD(cropped);

	// show non-SIMD output image
	namedWindow("Sobel-Non-SIMD", CV_WINDOW_AUTOSIZE);
	imshow("Sobel-Non-SIMD", imgOutputNonSIMD);

	/*********** SIMD **********/

	//imgInput = imread("boat.png", CV_LOAD_IMAGE_GRAYSCALE);
	imgOutputSIMD.create(imgInput.size(), imgInput.depth());

	__m128i p1, p2, p3, p4, p5, p6, p7, p8, p9;
	__m128i gx, gy, temp, G;

	outputPointer = imgOutputSIMD.ptr<uchar>();
	inputPointer = imgInput.ptr<uchar>();

	//width = imgInput.cols;
	//height = imgInput.rows;

	// get the current cpu time
	QueryPerformanceCounter(&t1); // start timer

	for (int i = (border - 1); i < height - (border + 1); i += 1) {
		for (int j = (border - 1); j < width - (2 * border - 1); j += 8) {

			/*
			Sobel operator input matrix
			+~~~~~~~~~~~~~~+
			| p1 | p2 | p3 |
			|~~~~+~~~~+~~~~+
			| p4 | p5 | p6 |
			|~~~~+~~~~+~~~~+
			| p7 | p8 | p9 |
			+~~~~+~~~~+~~~~+
			*/

			/*
			__m128i _mm_loadu_si128 (__m128i const* mem_addr)
			Load 128-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
			*/

			p1 = _mm_loadu_si128((__m128i*)(inputPointer + i * width + j));
			p2 = _mm_loadu_si128((__m128i*)(inputPointer + i * width + j + 1));
			p3 = _mm_loadu_si128((__m128i*)(inputPointer + i * width + j + 2));

			p4 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 1) * width + j));
			p5 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 1) * width + j + 1));
			p6 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 1) * width + j + 2));

			p7 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 2) * width + j));
			p8 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 2) * width + j + 1));
			p9 = _mm_loadu_si128((__m128i*)(inputPointer + (i + 2) * width + j + 2));

			/* 
			__m128i _mm_srli_epi16 (__m128i a, int imm8)
			Shift packed 16-bit integers in a right by imm8 while shifting in zeros, and store the Gs in dst.

			__m128i _mm_unpacklo_epi8 (__m128i a, __m128i b)
			Unpack and interleave 8-bit integers from the low half of a and b, and store the Gs in dst.
			*/

			// convert image 8-bit unsigned integer data to 16-bit signed integers to use in arithmetic operations
			p1 = _mm_srli_epi16(_mm_unpacklo_epi8(p1, p1), 8);
			p2 = _mm_srli_epi16(_mm_unpacklo_epi8(p2, p2), 8);
			p3 = _mm_srli_epi16(_mm_unpacklo_epi8(p3, p3), 8);
			p4 = _mm_srli_epi16(_mm_unpacklo_epi8(p4, p4), 8);
			p5 = _mm_srli_epi16(_mm_unpacklo_epi8(p5, p5), 8);
			p6 = _mm_srli_epi16(_mm_unpacklo_epi8(p6, p6), 8);
			p7 = _mm_srli_epi16(_mm_unpacklo_epi8(p7, p7), 8);
			p8 = _mm_srli_epi16(_mm_unpacklo_epi8(p8, p8), 8);
			p9 = _mm_srli_epi16(_mm_unpacklo_epi8(p9, p9), 8);

			/*
			__m128i _mm_add_epi16 (__m128i a, __m128i b)
			Add packed 16-bit integers in a and b, and store the Gs in dst.

			__m128i _mm_sub_epi16 (__m128i a, __m128i b)
			Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the Gs in dst.
			*/

			// Calculating Gx = (p3 + 2 * p6 + p9) - (p1 + 2 * p4 + p7)
			gx = _mm_add_epi16(p6, p6);		// 2*p6
			gx = _mm_add_epi16(gx, p3);		// p3 + 2*p6
			gx = _mm_add_epi16(gx, p9);		// p3 + 2*p6 + p9
			gx = _mm_sub_epi16(gx, p1);		// p3 + 2*p6 + p9 - p1
			temp = _mm_add_epi16(p4, p4);	// 2*p4
			gx = _mm_sub_epi16(gx, temp);	// p3 + 2*p6 + p9 - (p1 + 2*p4)
			gx = _mm_sub_epi16(gx, p7);		// p3 + 2*p6 + p9 - (p1 + 2*p4 + p7)

			// Calculating Gy = (p1 + 2 * p2 + p3) - (p7 + 2 * p8 + p9)
			gy = _mm_add_epi16(p2, p2);		// 2*p2
			gy = _mm_add_epi16(gy, p1);		// p1 + 2*p2
			gy = _mm_add_epi16(gy, p3);		// p1 + 2*p2 + p3
			gy = _mm_sub_epi16(gy, p7);		// p1 + 2*p2 + p3 - p7
			temp = _mm_add_epi16(p8, p8);	// 2*p8
			gy = _mm_sub_epi16(gy, temp);	// p1 + 2*p2 + p3 - (p7 + 2*p8)
			gy = _mm_sub_epi16(gy, p9);		// p1 + 2*p2 + p3 - (p7 + 2*p8 + p9)

			/*
			__m128i _mm_abs_epi16 (__m128i a)
			Compute the absolute value of packed 16-bit integers in a, and store the unsigned Gs in dst.
			*/

			gx = _mm_abs_epi16(gx); // |Gx|
			gy = _mm_abs_epi16(gy); // |Gy|

			// G = |Gx| + |Gy|
			G = _mm_add_epi16(gx, gy);

			/*
			__m128i _mm_packus_epi16 (__m128i a, __m128i b)
			Convert packed 16-bit integers from a and b to packed 8-bit integers using unsigned saturation, and store the results in dst.
			*/
			G = _mm_packus_epi16(G, G);

			/*
			void _mm_storeu_si128 (__m128i* mem_addr, __m128i a)
			Store 128-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
			*/
			_mm_storeu_si128((__m128i*)(outputPointer + (i + 1) * width + j + 1), G);

		}
	}

	// get the current cpu time
	QueryPerformanceCounter(&t2); // stop timer
	// calculate and print elapsed time in microseconds
	elapsedTime = static_cast<double>(t2.QuadPart - t1.QuadPart) * microseconds_per_count;
	cout << "Execution time for SIMD Sobel edge detection:" << endl;
	cout << elapsedTime << " us" << endl;

	// crop image and remove added borders
	imgOutputSIMD = imgOutputSIMD(cropped);

	// show SIMD output image
	namedWindow("Sobel-SIMD", CV_WINDOW_AUTOSIZE);
	imshow("Sobel-SIMD", imgOutputSIMD);

	/************* OpenCV Built-in Sobel *************/

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src;
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_8U;

	src = imgInput(cropped);

	// get the current cpu time
	QueryPerformanceCounter(&t1); // start timer

	// Gradient X
	Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x, 1, 0);

	// Gradient Y
	Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y, 1, 0);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 2.0, abs_grad_y, 2.0, 0, grad);

	// get the current cpu time
	QueryPerformanceCounter(&t2); // stop timer
	// calculate and print elapsed time in microseconds
	elapsedTime = static_cast<double>(t2.QuadPart - t1.QuadPart) * microseconds_per_count;
	cout << "Execution time for OpenCV Sobel edge detection:" << endl;
	cout << elapsedTime << " us" << endl;

	namedWindow("Sobel-OpenCV", CV_WINDOW_AUTOSIZE);
	imshow("Sobel-OpenCV", grad);

	waitKey(); // Wait for a keystroke in the window
	getchar();
	return 0;
}

//imgInput.copyTo(imgOutputNonSIMD);
//bitwise_not(imgInput, imgOutputNonSIMD);
