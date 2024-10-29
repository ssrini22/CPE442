/*******************************************************
* File: lab_4.h
*
* Description: Header file for Lab 4
*
* Author: Sanjeev Srinivasan
*
* Date: 10.22.2024
*
* Revision history: 1.0 - Include opencv, pthreads
*
********************************************************/

#ifndef LAB_4_H
#define LAB_4_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <pthread.h>


void* processQuarter(void* arg);
cv::Mat to442_grayscale(const cv::Mat& frame);
cv::Mat to442_sobel(const cv:: Mat& gray);
cv::Mat compute_gray_sobel(const cv:: Mat& arb);
int main(int argc, char** argv);

#endif

