/*******************************************************
* File: lab_6.h
*
* Description: Header file for Lab 6
*
* Author: Sanjeev Srinivasan
*
* Date: 11.27.2024
*
* Revision history: 1.0 - Include opencv, pthreads, neon
*
********************************************************/

#ifndef LAB_6_H
#define LAB_6_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <pthread.h>
#include <arm_neon.h>
#include <chrono>

void* processQuarter(void* arg);
cv::Mat to442_grayscale(const cv::Mat& frame);
cv::Mat neon_sobel(const cv::Mat& gray);
cv::Mat compute_gray_sobel(const cv:: Mat& arb);
int main(int argc, char** argv);

#endif


