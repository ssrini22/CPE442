/*********************************************************
* File: lab_3.cpp
*
* Description: Displays video with Sobel Filter
*
* Author: Sanjeev Srinivasan
*
* Revisions: 1.0 - Sobel Filter on Video
*
**********************************************************/

#include "lab_3.h"

/*-----------------------------------------------------
* Function: to442_grayscale(const cv::Mat *frame, cv::Mat *gray)
*
* Description: Turn OpenCV RGB Mat into grayscale Mat, using
* algorithm discussed in class 
*   
* parameter(s):
* cv:: Mat *frame: RGB frame matrix
* 
* return:
* cv:: Mat: grayscale matrix
*
*--------------------------------------------------------*/
cv::Mat to442_grayscale(const cv::Mat& frame) {
	cv::Mat gray(frame.rows, frame.cols, CV_8UC1);	
	for (int y = 0; y < frame.rows; y++){
		for (int x = 0; x < frame.cols; x++){
			cv::Vec3b pixel = frame.at<cv::Vec3b>(y,x);
			float grayValue = (0.2126f * pixel[2]) + (0.7152f * pixel[1]) + (0.0722f * pixel[0]);
			gray.at<uchar>(y,x) = static_cast<uchar>(grayValue);
		}	
	} 	
	return gray;
}

/*-----------------------------------------------------
* Function: void to442_sobel(const cv::Mat *gray, cv::Mat *sobel)
*
* Description: Applies the Sobel Filter
* 
* parameter(s):
* cv:: Mat *gray: grayscale matrix
* 
* return:
* cv:: Mat *sobel: sobel matrix
*
*--------------------------------------------------------*/
cv::Mat to442_sobel(const cv::Mat& gray) {
	cv::Mat sobel(gray.rows - 2, gray.cols - 2, CV_8UC1);
	int Gx[3][3] = {{-1,0,1},
					{-2,0,2},
					{-1,0,1}};
	int Gy[3][3] = {{1,2,1},
					{0,0,0},
					{-1,-2,-1}};
	for (int y = 1; y < gray.rows - 1; y++){
		for (int x = 1; x < gray.cols - 1; x++){
			int16_t sumX = 0;
			int16_t sumY = 0;

			for (int i = -1; i < 2; i++){
				for (int j = -1; j < 2; j++){
					sumX += gray.at<uchar>(y+i, x+j) * Gx[i+1][j+1];
					sumY += gray.at<uchar>(y+i, x+j) * Gy[i+1][j+1];
				}
			}
			int16_t mag = abs(sumX) + abs(sumY);
			if (mag > 255){
				mag = 255;
			}
			sobel.at<uchar>(y-1,x-1) = static_cast<uchar>(mag);
		}	
	}
	return sobel;
}

/*-----------------------------------------------------
* Function: main(int argc, char** argv)
*
* Description: Takes video path from command line,
* Applies sobel filter to video
*
* int argc: argument count
* char** argv: argument vector
*
* return: uint8_t
*--------------------------------------------------------*/
int main(int argc, char** argv) {
	if (argc != 2){
		std::cerr << "Argument Error" << std::endl;
		return -1;
	}

	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened()){
		std::cerr << "Video File Error" << std::endl;
		return -1;
	}

	cv::Mat frame, gray, sobel;
	
	while(true){
		cap >> frame;
		if (frame.empty()){
			break;
		}
		
		gray = to442_grayscale(frame);
		sobel = to442_sobel(gray);

		//cv::imshow("Original Frame", frame);
		//cv::imshow("Grayscale Frame", gray);
		cv::namedWindow("Sobel Frame", cv::WINDOW_NORMAL);
		cv::imshow("Sobel Frame", sobel);

		if (cv::waitKey(30) == 'q'){
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}

