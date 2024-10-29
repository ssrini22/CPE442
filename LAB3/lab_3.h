/*******************************************************
* File: lab_3.h
*
* Description: Header file for Lab 3
*
* Author: Sanjeev Srinivasan
*
* Date: 10.8.2024
*
* Revision history: 1.0 - Include opencv
*
********************************************************/

#ifndef LAB_3_H
#define LAB_3_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <algorithm>

cv::Mat to442_grayscale(const cv::Mat& frame);
cv::Mat to442_sobel(const cv:: Mat& gray);

#endif

