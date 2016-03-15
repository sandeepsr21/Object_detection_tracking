#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
int H_MIN = 26;
int S_MIN = 52;
int V_MIN = 14;
int H_MAX = 67;
int S_MAX = 256;
int V_MAX = 256;
void onTrackbarSlide(int, void*)
{    
}

/** @function main */
int main(int argc, char** argv)
{
	Mat src,hsv,mask;
	Mat rectErosione;
	Mat rectDilataz;
	Rect r;
	//video capture object to acquire webcam feed
	VideoCapture capture;
	vector<Vec3f> circles;
	Point p2;

	double ticks = 0;
	bool found = false;
	bool draw_path = false;
	double precTick = ticks;
	double dT = 0;
	int notFoundCount = 0;

	//Vector<Point> t20_frames;
	// >>>> Kalman Filter
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	int ntrackcount = 0;
	int ntrackframes = 30;
	
	//vector<Rect> boundRect1(ntrackframes);
	vector<Rect> boundRect1(ntrackframes);
	unsigned int type = CV_32F;
	Rect predRect;
	Point p1;
	p1.x = -1;
	p1.y = -1;
	Point Pt, Pt1;
	KalmanFilter kf(stateSize, measSize, contrSize, type);

	Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
										//cv::Mat procNoise(stateSize, 1, type)
										// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

										// Transition State Matrix A
										// Note: set dT at each processing step!
										// [ 1 0 dT 0  0 0 ]
										// [ 0 1 0  dT 0 0 ]
										// [ 0 0 1  0  0 0 ]
										// [ 0 0 0  1  0 0 ]
										// [ 0 0 0  0  1 0 ]
										// [ 0 0 0  0  0 1 ]
	setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));


	//open capture object at location zero (default location for webcam)
	capture.open(0);
	waitKey(0);


	/*namedWindow("trackbar", 0);

	//metodo che crea le trackbar(label, finestra, valore da cambiare, valore massimo,action listener)
	createTrackbar("H-min", "trackbar", &H_MIN, 256, onTrackbarSlide);
	createTrackbar("S-min", "trackbar", &S_MIN, 256, onTrackbarSlide);
	createTrackbar("V-min", "trackbar", &V_MIN, 256, onTrackbarSlide);
	createTrackbar("H-max", "trackbar", &H_MAX, 256, onTrackbarSlide);
	createTrackbar("S-max", "trackbar", &S_MAX, 256, onTrackbarSlide);
	createTrackbar("V-max", "trackbar", &V_MAX, 256, onTrackbarSlide);
	*/

	while (1) {
		//store image to matrix
		capture.read(src);
		if (!src.data)
		{
			return -1;
		}

		ticks = (double)cv::getTickCount();

		dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

		if (found)
		{
			// >>>> Matrix A
			kf.transitionMatrix.at<float>(2) = dT;
			kf.transitionMatrix.at<float>(9) = dT;
			// <<<< Matrix A

			//cout << "dT:" << endl << dT << endl;

			state = kf.predict();
			//	cout << "State post:" << endl << state << endl;
	
			
			predRect.width = state.at<float>(4);
			predRect.height = state.at<float>(5);
			predRect.x = state.at<float>(0) - predRect.width / 2;
			predRect.y = state.at<float>(1) - predRect.height / 2;

			rectangle(src,
				Point(r.x, r.y),
				Point(r.height, r.width),
				(0, 198, 255),
				+1,
				4);

			if (draw_path)
			{
				if (ntrackcount == ntrackframes)
					ntrackcount = 0;

				if (ntrackcount > 0 && ntrackcount < ntrackframes)
				{
					for (int iTrack = 0; iTrack < ntrackcount - 1; iTrack++)
					{

						Pt.x = boundRect1[iTrack].x;
						Pt.y = boundRect1[iTrack].y;
						Pt1.x = boundRect1[iTrack + 1].x;;
						Pt1.y = boundRect1[iTrack + 1].y;
						
						line(src, Pt, Pt1, Scalar(0, 0, 255), 2); // Corrected                                
					}
				}
				if (ntrackcount == ntrackframes)
					ntrackcount = 0;
			}
		
			//center.x = state.at<float>(0);
			//center.y = state.at<float>(1);
		}


		cvtColor(src, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), mask);
	//	inRange(hsv, Scalar(29, 86, 6), Scalar(64, 255, 255), mask);
		//inRange(hsv, Scalar(26, 75, 67), Scalar(256, 256, 256), mask);
	    /// Convert it to gray
		
		rectErosione = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(mask, mask, rectErosione);
		erode(mask, mask, rectErosione);
		erode(mask, mask, rectErosione);

		rectDilataz = getStructuringElement(MORPH_RECT, Size(8, 8));
		dilate(mask, mask, rectDilataz);
		dilate(mask, mask, rectDilataz);
		dilate(mask, mask, rectDilataz);

		/// Reduce the noise so we avoid false circle detection
		GaussianBlur(mask, mask, Size(11, 11), 2, 2);

	

		/// Apply the Hough Transform to find the circles
		//HoughCircles(mask, circles, CV_HOUGH_GRADIENT, 1, mask.rows / 20, 100, 100, 0, 0);
		HoughCircles(mask, circles, CV_HOUGH_GRADIENT, 2, mask.rows / 4, 100, 40, 10, 120);
		/// Draw the circles detected

			for (size_t i = 0; i < circles.size(); i++)
			{
				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]);
				// circle center
				circle(src, center, 3, Scalar(0, 0, 255), -1, 8, 0);
				// circle outline
				//circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
				//t20_frames.size() = circles.size();
				//p2.x = center.x;
				//p2.y = center.y;
				boundRect1[ntrackcount].x = center.x;
				boundRect1[ntrackcount].y = center.y;
				r.x = center.x - (radius + 10);
				r.y = center.y - (radius + 10);
				r.height = (center.x) + (radius + 10);
				r.width = (center.y) + (radius + 10);
				draw_path = true;

				/*	rectangle(src,
						cvPoint(center.x-(radius+10),center.y-(radius+10)),
						cvPoint((center.x) + (radius + 10), (center.y) + (radius + 10)),
						(0, 198, 255),
						+1,
						4);*/

			}
			
			ntrackcount++;



		if (circles.size() == 0)
		{
			notFoundCount++;
			draw_path = false;
			//cout << "notFoundCount:" << notFoundCount << endl;
			if (notFoundCount >= 100)
			{
				found = false;
			}
			/*else
			kf.statePost = state;*/
		}
		else
		{
			notFoundCount = 0;

			meas.at<float>(0) = r.x + r.width / 2;
			meas.at<float>(1) = r.y + r.height / 2;
			meas.at<float>(2) = (float)r.width;
			meas.at<float>(3) = (float)r.height;

			if (!found) // First detection!
			{
				// >>>> Initialization
				kf.errorCovPre.at<float>(0) = 1; // px
				kf.errorCovPre.at<float>(7) = 1; // px
				kf.errorCovPre.at<float>(14) = 1;
				kf.errorCovPre.at<float>(21) = 1;
				kf.errorCovPre.at<float>(28) = 1; // px
				kf.errorCovPre.at<float>(35) = 1; // px

				state.at<float>(0) = meas.at<float>(0);
				state.at<float>(1) = meas.at<float>(1);
				state.at<float>(2) = 0;
				state.at<float>(3) = 0;
				state.at<float>(4) = meas.at<float>(2);
				state.at<float>(5) = meas.at<float>(3);
				// <<<< Initialization

				found = true;
			}
			else
				kf.correct(meas); // Kalman Correction


			
								  //			cout << "Measure matrix:" << endl << meas << endl;
		}
	/// Show your results
		//namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
		imshow("Demo", src);
		//imshow("hsv", hsv);
		imshow("mask", mask);
		waitKey(1);
	}
	return 0;
}