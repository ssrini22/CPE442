/*********************************************************
* File: lab_6.cpp
*
* Description: Optimized Sobel with perf
*
* Author: Sanjeev Srinivasan
*
* Revisions: 1.0 - Sobel Filter on Video with Threading
* and Vector Intrinsics and Performance
*
**********************************************************/

#include "lab_6.h"

pthread_t thread[4];
cv::Mat frame;
cv::Mat result;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/*-----------------------------------------------------
* Function: *void processQuarter(void* arg)
*
* Description: Takes an int corresponding to the quarter
* of the image to be processed and applies filter
*   
* parameter(s):
* *void arg: integer corresponding to quarter
*
*--------------------------------------------------------*/
void* processQuarter(void* arg){
	//calculate row params
	int quarter = *(int*) arg;
	int rows = frame.rows;
	int cols = frame.cols;
	int q = rows / 4;
   	int start_row = quarter * q;
	int end_row = (quarter == 3) ? (rows) : ((quarter + 1) * q) + 1;
	int result_start_row = start_row;
	int result_end_row = end_row;

	//get and process part of frame
	cv::Mat color = frame(cv::Range(start_row, end_row), cv::Range::all());
    cv::Mat	process = compute_gray_sobel(color);	
	//std::cout << "PROCESS" << quarter << "size " << process.size() << std::endl;
	//lock when writing to result Mat
	pthread_mutex_lock(&mutex);
	try{
		process.copyTo(result(cv::Range(result_start_row, result_end_row), cv::Range::all()));
	}
	catch (const cv::Exception& e){
		std::cerr << "Thread " << quarter << "INV " << result_start_row << " : " 
			<< result_end_row << " size " << result.rows << std::endl;
		std::cerr << "L " << e.line << " F " << e.func << std::endl;
	}
	
	pthread_mutex_unlock(&mutex);
//    std::cout << "T" << quarter << " RESULT" << result.size() << std::endl;
//	std::cout << "Thread " << quarter << "RROWS " << result_start_row << " : " 
//			<< result_end_row << " size " << result.rows << std::endl;

	return nullptr;
}


/*-----------------------------------------------------
* Function: cv::Mat to442_grayscale(const cv::Mat& frame)
*
* Description: Turn OpenCV RGB Mat into grayscale Mat, using
* algorithm discussed in class 
*   
* parameter(s):
* cv:: Mat& frame: RGB frame matrix
* 
* return:
* cv:: Mat: grayscale matrix
*
*--------------------------------------------------------*/
cv::Mat to442_grayscale(const cv::Mat& frame) {
	cv::Mat gray(frame.rows, frame.cols, CV_8UC1);

    const int redWeight = static_cast<int>(0.2126f * 1024);
    const int greenWeight = static_cast<int>(0.7152f * 1024);
    const int blueWeight = static_cast<int>(0.0722f * 1024);

    cv::Mat weights = (cv::Mat_<float>(1, 3) <<
                        blueWeight/1024.0f, greenWeight/1024.0f, redWeight/1024.0f);
    cv::transform(frame, gray, weights);

	return gray;
}


/*-----------------------------------------------------
* Function: cv::Mat neon_sobel(const cv::Mat& gray)
*
* Description: Applies the Sobel Filter with Vector
* intrinsics
* 
* parameter(s):
* cv:: Mat& gray: grayscale matrix
* 
* return:
* cv:: Mat sobel: sobel matrix
*
*--------------------------------------------------------*/
cv::Mat neon_sobel(const cv::Mat& gray) {
    int rows = gray.rows;
    int cols = gray.cols;

    cv::Mat pad;
    cv::copyMakeBorder(gray, pad, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    
    //std::cout << rows << ":" << cols << " P " << pad_rows << ":" << pad_cols << std::endl;	
    cv::Mat sobel(rows, cols, CV_8UC1);

    for (int y = 1; y < rows; y++){
        const uint8_t* row0 = pad.ptr<uint8_t>(y - 1);
        const uint8_t* row1 = pad.ptr<uint8_t>(y);
        const uint8_t* row2 = pad.ptr<uint8_t>(y + 1);
        uint8_t* row_sobel = sobel.ptr<uint8_t>(y - 1);
        
        for (int x = 0; x <= cols - 8; x+=8){
            uint8x8_t r0 = vld1_u8(row0 + x); 
            uint8x8_t r1 = vld1_u8(row1 + x);
            uint8x8_t r2 = vld1_u8(row2 + x);

            uint16x8_t r0_16 = vmovl_u8(r0);
            uint16x8_t r1_16 = vmovl_u8(r1);
            uint16x8_t r2_16 = vmovl_u8(r2);

            uint16x8_t r0_left = vextq_u16(r0_16, r0_16, 7); 
            uint16x8_t r0_right = vextq_u16(r0_16, r0_16, 1); 
       
            uint16x8_t r1_left = vextq_u16(r1_16, r1_16, 7); 
            uint16x8_t r1_right = vextq_u16(r1_16, r1_16, 1); 
        
            uint16x8_t r2_left = vextq_u16(r2_16, r2_16, 7); 
            uint16x8_t r2_right = vextq_u16(r2_16, r2_16, 1);
            
            int16x8_t gx = vsubq_s16(vreinterpretq_s16_u16(r0_right), vreinterpretq_s16_u16(r0_left));
            gx = vmlaq_n_s16(gx, vreinterpretq_s16_u16(r1_right), 2);
            gx = vmlsq_n_s16(gx, vreinterpretq_s16_u16(r1_left), 2);
            gx = vmlaq_n_s16(gx, vreinterpretq_s16_u16(r2_right), 1);
            gx = vmlsq_n_s16(gx, vreinterpretq_s16_u16(r2_left), 1);

            int16x8_t gy = vsubq_s16(vreinterpretq_s16_u16(r0_left), vreinterpretq_s16_u16(r2_left));
            gy = vmlaq_n_s16(gy, vreinterpretq_s16_u16(r0_16), 2);
            gy = vmlaq_n_s16(gy, vreinterpretq_s16_u16(r0_right), 1);
            gy = vmlsq_n_s16(gy, vreinterpretq_s16_u16(r2_16), 2);
            gy = vmlsq_n_s16(gy, vreinterpretq_s16_u16(r2_right), 1);

            int16x8_t abs_gx = vabsq_s16(gx);
            int16x8_t abs_gy = vabsq_s16(gy);
            int16x8_t mag = vaddq_s16(abs_gx, abs_gy);

            uint8x8_t result = vqmovn_u16(vreinterpretq_u16_s16(mag));
            vst1_u8(row_sobel + x, result);
        }
    }

    return sobel;
}

/*-----------------------------------------------------
* Function: void compute_gray_sobel(const cv::Mat& arb)
*
* Description: Applies grayscale and sobel to arbitrairily
* sized Mat
* 
* parameter(s):
* cv:: Mat& arb: frame matrix
* 
* return:
* cv:: Mat sobel: sobel matrix
*
*--------------------------------------------------------*/
cv::Mat compute_gray_sobel(const cv::Mat& arb) {
	if (arb.empty()){
		return cv::Mat();
	}
	cv::Mat gray, sobel;
	gray = to442_grayscale(arb);
	sobel = neon_sobel(gray);
    //std::cout << "COMPSOBEL" << sobel.size() << std::endl;
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
	//init attr
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	if (argc != 2){
		std::cerr << "Argument Error" << std::endl;
		return -1;
	}

	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened()){
		std::cerr << "Video File Error" << std::endl;
		return -1;
	}

    int frameCount = 0;
    auto start = std::chrono::high_resolution_clock::now();

	while(true){
		cap >> frame;
		if (frame.empty()){
			break;
		}
		
		result = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		//create threads
        int args[4];
		for (int i = 0; i < 4; i++){
		    args[i] = i;
			pthread_create(&thread[i], NULL, processQuarter, &args[i]);
		}
		std::cout << "CREATED ALL" << std::endl;
		
		//join threads
		for (int i = 0; i < 4; i++){
			pthread_join(thread[i], nullptr);
		}
		//display frame
		std::cout << "DISPLAY?" << std::endl;
		cv::namedWindow("Sobel Frame", cv::WINDOW_NORMAL);
		cv::imshow("Sobel Frame", result);
		if (cv::waitKey(30) == 'q'){
			break;
		}

        frameCount++;
        
        if (frameCount % 30 == 0){
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            double fps = frameCount/elapsed;

            std::cout << "FPS: " << fps << std::endl;

            frameCount = 0;
            start = std::chrono::high_resolution_clock::now();
        }

	}
	cap.release();
	cv::destroyAllWindows();

	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&mutex);
	
	return 0;
}


