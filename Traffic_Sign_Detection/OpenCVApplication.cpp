// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <stack>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the �diblook style�
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


bool isInsideB(const Mat_<Vec3b>& src, int x, int y) {
	return x >= 0 && x < src.rows && y >= 0 && y < src.cols;
}

double axisOfElongation(const Mat_<Vec3b>& src, Vec3b pixel, int ri, int ci) {
	double numarator = 0.0;
	double numitor = 0.0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == pixel) {
				numarator += 2.0 * (i - ri) * (j - ci);
				numitor += (j - ci) * (j - ci) - (i - ri) * (i - ri);
			}
		}
	}
	double axisAngle = atan2(numarator, numitor) / 2.0;
	return axisAngle;
}


double thinnessRatio(int area, int perimeter) {
	double T = 4 * PI * ((double)area / (perimeter * perimeter));
	return T;
}
int areaPlusCentereOfMass(const Mat_<Vec3b>& src, Vec3b pixel, int* ri, int* ci) {
	int rSum = 0;
	int cSum = 0;
	int area = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == pixel) {
				area++;
				rSum += i;
				cSum += j;
			}
		}
	}

	*ri = rSum / area;
	*ci = cSum / area;
	return area;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// COLOR BASED DETECTION - OPRUTA GEORGE////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////// 

bool perimeterPixel2(const Mat_<Vec3b>& src, Vec3b pixel, int x, int y) {
	return (isInsideB(src, x + 1, y + 1) && src(x + 1, y + 1) != pixel) ||
		(isInsideB(src, x, y + 1) && src(x, y + 1) != pixel) ||
		(isInsideB(src, x - 1, y + 1) && src(x - 1, y + 1) != pixel) ||
		(isInsideB(src, x - 1, y) && src(x - 1, y) != pixel) ||
		(isInsideB(src, x - 1, y - 1) && src(x - 1, y - 1) != pixel) ||
		(isInsideB(src, x, y - 1) && src(x, y - 1) != pixel) ||
		(isInsideB(src, x + 1, y - 1) && src(x + 1, y - 1) != pixel) ||
		(isInsideB(src, x + 1, y) && src(x + 1, y) != pixel);
}

int perimeter2(const Mat_<Vec3b>& src, Vec3b pixel, Mat_<Vec3b>& dst) {
	int perimeter = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == pixel) {
				if (perimeterPixel2(src, pixel, i, j)) {
					perimeter++;
					dst(i, j) = pixel;
				}
			}
		}
	}

	return perimeter;
}

Mat normalizeColor(Mat src, Vec3b color) {

	Mat dst(src.rows, src.cols, CV_8UC3, cv::Scalar(0));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
				dst.at<Vec3b>(i, j) = color;
			}
		}
	}

	return dst;


}

Mat erosion(Mat src) {
	int i, j, k;
	int threshold = 127; // set threshold at desired value
	int di[8] = { -1,  0, 1, 0, -1, -1, 1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };

	int height = src.rows;
	int width = src.cols;

	// image binarization

	Mat img(height, width, CV_8UC1);
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) < threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}

	Mat dst = Mat(height, width, CV_8UC1, cv::Scalar(0));

	int n = 8; // pentru a selecta vecinatatea - putem alege vecinatatea de 8

	for (i = 1; i < height - 1; i++) {
		for (j = 1; j < width - 1; j++) {  // Adjusted loop bounds
			if (img.at<uchar>(i, j) == 255) {
				// Check neighbors
				bool isBorder = true;
				for (k = 0; k < n; k++) {
					int ni = i + di[k];
					int nj = j + dj[k];
					if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
						if (img.at<uchar>(ni, nj) != 255) {
							isBorder = false;

						}
					}
				}
				if (isBorder) {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return dst;
}

void signDetectoion() {

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_UNCHANGED);
		Mat original = src.clone();

		if (src.empty()) {
			std::cout << "Error loading image" << std::endl;
			return;
		}

		Mat H = Mat(src.rows, src.cols, CV_8UC1);
		Mat S = Mat(src.rows, src.cols, CV_8UC1);
		Mat V = Mat(src.rows, src.cols, CV_8UC1);
		Vec3b pixel;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				pixel = src.at<Vec3b>(i, j);

				float b = (float)pixel[0] / 255.0f;
				float g = (float)pixel[1] / 255.0f;
				float r = (float)pixel[2] / 255.0f;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);
				float C = M - m;

				float Vvalue = M;
				float Svalue = (Vvalue == 0.0f) ? 0.0f : (C / Vvalue);
				float Hvalue = 0.0f;

				if (C != 0.0f) {
					if (M == r) {
						Hvalue = 60.0f * (g - b) / C;
					}
					else if (M == g) {
						Hvalue = 120.0f + 60.0f * (b - r) / C;
					}
					else if (M == b) {
						Hvalue = 240.0f + 60.0f * (r - g) / C;
					}
				}

				if (Hvalue < 0.0f) {
					Hvalue += 360.0f;
				}

				H.at<uchar>(i, j) = static_cast<uchar>(Hvalue * 255.0f / 360.0f);
				S.at<uchar>(i, j) = static_cast<uchar>(Svalue * 255.0f);
				V.at<uchar>(i, j) = static_cast<uchar>(Vvalue * 255.0f);
			}
		}


		Mat hsv;
		std::vector<Mat> hsv_channels = { H, S, V };
		merge(hsv_channels, hsv);

		Scalar lower_red1(0, 50, 50);
		Scalar upper_red1(35, 255, 255);

		cv::Scalar lower_red2(155, 100, 100); // Additional lower bound to handle red hues around 0 degrees
		cv::Scalar upper_red2(190, 255, 255); // Additional upper bound to handle red hues around 0 degrees

		Mat maskR1;
		Mat maskR2;
		inRange(hsv, lower_red1, upper_red1, maskR1);
		inRange(hsv, lower_red2, upper_red2, maskR2);

		Mat maskR;
		maskR = maskR1 + maskR2;

		//Mat maskRFiltered;
		//maskRFiltered= erosion(maskR);

		Mat resultR;
		bitwise_and(src, src, resultR, maskR);

		resultR = normalizeColor(resultR, Vec3b(0, 0, 255));

		Mat_<Vec3b> perimeterR(src.rows, src.cols, Vec3b(0, 0, 0));
		Mat_<Vec3b> redClone = resultR.clone();

		int pR = perimeter2(redClone, Vec3b(0, 0, 255), perimeterR);

		int riR = 0;
		int ciR = 0;
		int areaR = areaPlusCentereOfMass(redClone, Vec3b(0, 0, 255), &riR, &ciR);
		double TR = thinnessRatio(areaR, pR);

		float redRatio = (float)areaR / (src.rows * src.cols);

		double angleR = axisOfElongation(redClone, Vec3b(0, 0, 255), riR, ciR);
		double end_rR = riR + 100 * sin(angleR);
		double end_cR = ciR + 100 * sin(angleR);


		Scalar lower_blue(80, 50, 50);
		Scalar upper_blue(130, 255, 255);

		Mat maskB;
		inRange(hsv, lower_blue, upper_blue, maskB);


		Mat resultB;
		bitwise_and(src, src, resultB, maskB);

		resultB = normalizeColor(resultB, Vec3b(255, 0, 0));


		Mat_<Vec3b> perimeterB(src.rows, src.cols, Vec3b(0, 0, 0));
		Mat_<Vec3b> blueClone = resultB.clone();
		int pB = perimeter2(blueClone, Vec3b(255, 0, 0), perimeterB);

		int riB = 0;
		int ciB = 0;
		int areaB = areaPlusCentereOfMass(blueClone, Vec3b(255, 0, 0), &riB, &ciB);
		int TB = thinnessRatio(areaB, pB);

		float blueRatio = (float)areaB / (src.rows * src.cols);

		double angleB = axisOfElongation(blueClone, Vec3b(0, 0, 255), riB, ciB);
		double end_rB = riB + 100 * sin(angleB);
		double end_cB = ciB + 100 * sin(angleB);

		Scalar lower_yellow(60, 50, 50);
		Scalar upper_yellow(70, 255, 255);


		Mat maskY;
		inRange(hsv, lower_yellow, upper_yellow, maskY);

		Mat resultY;
		bitwise_and(src, src, resultY, maskY);

		resultY = normalizeColor(resultY, Vec3b(0, 255, 255));

		Mat_<Vec3b> perimeterY(src.rows, src.cols, Vec3b(0, 0, 0));
		Mat_<Vec3b> yellowClone = resultY.clone();

		int pY = perimeter2(yellowClone, Vec3b(0, 255, 255), perimeterY);

		int riY = 0;
		int ciY = 0;
		int areaY = areaPlusCentereOfMass(yellowClone, Vec3b(0, 255, 255), &riY, &ciY);
		int TY = thinnessRatio(areaY, pY);

		float yellowRatio = (float)areaY / (src.rows * src.cols);

		double angleY = axisOfElongation(yellowClone, Vec3b(0, 0, 255), riY, ciY);
		double end_rY = riY + 100 * sin(angleY);
		double end_cY = ciY + 100 * sin(angleY);


		Mat_<Vec3b> signPerimeter;


		float mostPixels = max(max(yellowRatio, blueRatio), redRatio);
		int whatColor = -1;
		if (mostPixels == yellowRatio) {
			whatColor = 0;
		}
		if (mostPixels == redRatio) {
			whatColor = 2;
		}
		if (mostPixels == blueRatio) {
			whatColor = 1;

		}

		switch (whatColor) {
		case 0: {
			signPerimeter = perimeterY.clone();
			int ri = riY;
			int ci = ciY;
			int endr = end_rY;
			int endc = end_cY;
			int differenceX = endc - ci;
			int differenceY = endr - ri;
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (signPerimeter(i, j) == Vec3b(0, 255, 255)) {
						src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

					}
				}
			}
			std::cout << "Based on yellow predominance looks like a warning sign\n";
			line(src, Point(ci - differenceY, ri - differenceX), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri - differenceX), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri + differenceX), Point(ci, ri + differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci + differenceY, ri + differenceX), Vec3b(0, 255, 0));

			line(src, Point(ci - differenceY, ri + differenceX), Point(ci - differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci - differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri + differenceX), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));
			break;
		}
		case 1: {
			signPerimeter = perimeterB.clone();
			int ri = riB;
			int ci = ciB;
			int endr = end_rB;
			int endc = end_cB;
			int differenceX = endc - ci;
			int differenceY = endr - ri;

			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (signPerimeter(i, j) == Vec3b(255, 0, 0)) {
						src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

					}
				}
			}
			std::cout << "Based on blue predominance looks like a mandatory sign\n";

			line(src, Point(ci - differenceY, ri - differenceX), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri - differenceX), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri + differenceX), Point(ci, ri + differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci + differenceY, ri + differenceX), Vec3b(0, 255, 0));

			line(src, Point(ci - differenceY, ri + differenceX), Point(ci - differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci - differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri + differenceX), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));
			break;

		}
		case 2: {
			signPerimeter = perimeterR.clone();
			int ri = riR;
			int ci = ciR;

			int endr = end_rR;
			int endc = end_cR;
			int differenceX = endc - ci;
			int differenceY = endr - ri;

			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (signPerimeter(i, j) == Vec3b(0, 0, 255)) {
						src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

					}
				}
			}
			std::cout << "Based on red predominance looks like a prohibitory or a warning sign \n";
			line(src, Point(ci - differenceY, ri - differenceX), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri - differenceX), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri + differenceX), Point(ci, ri + differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci + differenceY, ri + differenceX), Vec3b(0, 255, 0));

			line(src, Point(ci - differenceY, ri + differenceX), Point(ci - differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci - differenceY, ri), Point(ci - differenceY, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci, ri + differenceX), Point(ci, ri), Vec3b(0, 255, 0));
			line(src, Point(ci, ri), Point(ci, ri - differenceX), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri + differenceX), Point(ci + differenceY, ri), Vec3b(0, 255, 0));
			line(src, Point(ci + differenceY, ri), Point(ci + differenceY, ri - differenceX), Vec3b(0, 255, 0));

			break;
		}
		default: {
			break;
		}


		}

		namedWindow("Detected Sign", WINDOW_NORMAL);
		imshow("Detected Sign", src);
		namedWindow("Original Image", WINDOW_NORMAL);
		imshow("Original Image", original);

		namedWindow("ConturR?", WINDOW_NORMAL);
		imshow("ConturR?", perimeterR);

		namedWindow("ConturB?", WINDOW_NORMAL);
		imshow("ConturB?", perimeterB);

		namedWindow("ConturY?", WINDOW_NORMAL);
		imshow("ConturY?", perimeterY);

		namedWindow("MaksRed", WINDOW_NORMAL);
		imshow("MaksRed", maskR);
		namedWindow("CLONA R", WINDOW_NORMAL);
		imshow("CLONA R", redClone);
		namedWindow("Clona B", WINDOW_NORMAL);
		imshow("Clona B", blueClone);
		namedWindow("Clona Y", WINDOW_NORMAL);
		imshow("Clona Y", yellowClone);
		namedWindow("Detected Color: Red", WINDOW_NORMAL);
		imshow("Detected Color: Red", resultR);
		namedWindow("Detected Color: Blue", WINDOW_NORMAL);
		imshow("Detected Color: Blue", resultB);
		namedWindow("Detected Color: Green", WINDOW_NORMAL);
		imshow("Detected Color: Green", resultY);

		waitKey(0);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// CANNY BASED DETECTION - OANA SABAU //////////////////////////// 
///////////////////////////////////////////////////////////////////////////////////////////////////// 

int* calculateHistogramFromGivenImage(Mat img) {
	int i, j;

	int* histogram = (int*)calloc(256, sizeof(int));

	int height = img.rows;
	int width = img.cols;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			uchar val = img.at<uchar>(i, j);
			histogram[val]++;
		}
	}
	return histogram;
}

bool isInside(Mat img, int i, int j) {
	int rows = img.rows;
	int cols = img.cols;
	if (i >= 0 && i < rows && j >= 0 && j < cols) {
		return true;
	}
	return false;
}

Mat convolution(Mat src, Mat_<float> kernel) {

	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	int i, j, u, v;
	int k = kernel.rows / 2;
	float sumNegKernel = 0.0, sumPosKernel = 0.0;

	// determinare suma coeficientilor pozitivi si negativi
	// daca filtrul este trece-jos, nu exista valori negative in kernel

	for (i = 0; i < kernel.rows; i++) {
		for (j = 0; j < kernel.cols; j++)
		{
			if (kernel(i, j) < 0) {
				sumNegKernel += kernel(i, j);
			}
			if (kernel(i, j) > 0) {
				sumPosKernel += kernel(i, j);
			}
		}
	}

	float imin = 255.0 * sumNegKernel;
	float imax = 255.0 * sumPosKernel;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			float sum = 0.0;
			for (u = 0; u < kernel.rows; u++) {
				for (v = 0; v < kernel.cols; v++) {
					if (isInside(src, i + u - k, j + v - k)) {
						sum += kernel(u, v) * src.at<uchar>(i + u - k, j + v - k);
					}
				}
			}
			// aplicare formula si asigurare ca ne aflam in intervalul [0, 255]
			sum = ((sum - imin) / (imax - imin)) * 255.0;  // daca filtrul este trece-jos, ramane sum/c deoarece se simplifica constantele, iar imin este 0
			if (sum > 255.0) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				if (sum < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = sum;
				}
			}
		}
	}
	return dst;
}


std::pair<Mat_<float>, Mat_<float>> defineGradientComponents(const char* name) {

	Mat_<float> fx = Mat(3, 3, CV_32FC1);
	Mat_<float> fy = Mat(3, 3, CV_32FC1);

	if (strcmp("Prewitt", name) == 0) {

		// define x component
		fx(0, 0) = -1.0;
		fx(0, 1) = 0.0;
		fx(0, 2) = 1.0;
		fx(1, 0) = -1.0;
		fx(1, 1) = 0.0;
		fx(1, 2) = 1.0;
		fx(2, 0) = -1.0;
		fx(2, 1) = 0.0;
		fx(2, 2) = 1.0;

		// define y component
		fy(0, 0) = 1.0;
		fy(0, 1) = 1.0;
		fy(0, 2) = 1.0;
		fy(1, 0) = 0.0;
		fy(1, 1) = 0.0;
		fy(1, 2) = 0.0;
		fy(2, 0) = -1.0;
		fy(2, 1) = -1.0;
		fy(2, 2) = -1.0;
	}

	if (strcmp("Sobel", name) == 0) {

		// define x component
		fx(0, 0) = -1.0;
		fx(0, 1) = 0.0;
		fx(0, 2) = 1.0;
		fx(1, 0) = -2.0;
		fx(1, 1) = 0.0;
		fx(1, 2) = 2.0;
		fx(2, 0) = -1.0;
		fx(2, 1) = 0.0;
		fx(2, 2) = 1.0;

		// define y component
		fy(0, 0) = 1.0;
		fy(0, 1) = 2.0;
		fy(0, 2) = 1.0;
		fy(1, 0) = 0.0;
		fy(1, 1) = 0.0;
		fy(1, 2) = 0.0;
		fy(2, 0) = -1.0;
		fy(2, 1) = -2.0;
		fy(2, 2) = -1.0;
	}

	if (strcmp("Roberts", name) == 0) {

		fx = Mat_<float>(2, 2, CV_32FC1);
		fy = Mat_<float>(2, 2, CV_32FC1);

		// define x component
		fx(0, 0) = 1.0;
		fx(0, 1) = 0.0;
		fx(1, 0) = 0.0;
		fx(1, 1) = -1.0;

		// define y component
		fy(0, 0) = 0.0;
		fy(0, 1) = -1.0;
		fy(1, 0) = 1.0;
		fy(1, 1) = 0.0;
	}
	return { fx, fy };
}

// aceeasi functie, insa avem nevoie de valori de tip int, nu float
std::pair<Mat_<int>, Mat_<int>> defineGradientComponentsSobel(const char* name) {

	Mat_<int> fx = Mat(3, 3, CV_32SC1);
	Mat_<int> fy = Mat(3, 3, CV_32SC1);

	if (strcmp("Prewitt", name) == 0) {

		// define x component
		fx(0, 0) = -1;
		fx(0, 1) = 0;
		fx(0, 2) = 1;
		fx(1, 0) = -1;
		fx(1, 1) = 0;
		fx(1, 2) = 1;
		fx(2, 0) = -1;
		fx(2, 1) = 0;
		fx(2, 2) = 1;

		// define y component
		fy(0, 0) = 1;
		fy(0, 1) = 1;
		fy(0, 2) = 1;
		fy(1, 0) = 0;
		fy(1, 1) = 0;
		fy(1, 2) = 0;
		fy(2, 0) = -1;
		fy(2, 1) = -1;
		fy(2, 2) = -1;
	}

	if (strcmp("Sobel", name) == 0) {

		// define x component
		fx(0, 0) = -1;
		fx(0, 1) = 0;
		fx(0, 2) = 1;
		fx(1, 0) = -2;
		fx(1, 1) = 0;
		fx(1, 2) = 2;
		fx(2, 0) = -1;
		fx(2, 1) = 0;
		fx(2, 2) = 1;

		// define y component
		fy(0, 0) = 1;
		fy(0, 1) = 2;
		fy(0, 2) = 1;
		fy(1, 0) = 0;
		fy(1, 1) = 0;
		fy(1, 2) = 0;
		fy(2, 0) = -1;
		fy(2, 1) = -2;
		fy(2, 2) = -1;
	}

	if (strcmp("Roberts", name) == 0) {

		fx = Mat_<int>(2, 2, CV_32SC1);
		fy = Mat_<int>(2, 2, CV_32SC1);

		// define x component
		fx(0, 0) = 1;
		fx(0, 1) = 0;
		fx(1, 0) = 0;
		fx(1, 1) = -1;

		// define y component
		fy(0, 0) = 0;
		fy(0, 1) = -1;
		fy(1, 0) = 1;
		fy(1, 1) = 0;
	}
	return { fx, fy };
}


void applyKernelToSeeGradients(Mat src, const char* name) {
	std::pair<Mat_<float>, Mat_<float>> gradients = defineGradientComponents(name);
	Mat dst_x = convolution(src, gradients.first);
	Mat dst_y = convolution(src, gradients.second);

	imshow(std::string(name) + " gradient x", dst_x);
	imshow(std::string(name) + " gradient y", dst_y);
}


// aceeasi functie, doar ca returneaza gradientii
std::pair<Mat_<float>, Mat_<float>> applyKernel(Mat src, const char* name) {
	std::pair<Mat_<float>, Mat_<float>> gradients = defineGradientComponents(name);
	Mat dst_x = convolution(src, gradients.first);
	Mat dst_y = convolution(src, gradients.second);
	return { dst_x, dst_y };
}

Mat magnitude(Mat src, const char* name) {

	int i, j;
	Mat magnitude = Mat(src.rows, src.cols, CV_32SC1);

	std::pair<Mat_<float>, Mat_<float>> gradients = applyKernel(src, name);
	Mat_<float> gradient_X = gradients.first;
	Mat_<float> gradient_Y = gradients.second;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			magnitude.at<int>(i, j) = (int)(sqrt(gradient_X(i, j) * gradient_X(i, j) + gradient_Y(i, j) * gradient_Y(i, j)));
		}
	}
	return magnitude;
}

Mat direction(Mat src, const char* name) {

	int i, j;
	Mat direction = Mat(src.rows, src.cols, CV_32SC1);

	std::pair<Mat_<int>, Mat_<int>> gradients = applyKernel(src, name);
	Mat_<float> gradient_X = gradients.first;
	Mat_<float> gradient_Y = gradients.second;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			direction.at<int>(i, j) = (int)((atan2(gradient_Y(i, j), gradient_X(i, j)) * 180) / PI);

		}
	}
	if (strcmp("Roberts", name) == 0) {
		direction += 135;
	}
	return direction;
}

void applyGradient() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		std::pair<Mat_<float>, Mat_<float>> dst;

		applyKernelToSeeGradients(src, "Prewitt");
		applyKernelToSeeGradients(src, "Sobel");
		applyKernelToSeeGradients(src, "Roberts");

		Mat magNormalizedPrewitt, magNormalizedSobel, magNormalizedRoberts;
		Mat dirNormalizedPrewitt, dirNormalizedSobel, dirNormalizedRoberts;

		Mat magPrewitt = magnitude(src, "Prewitt");
		normalize(magPrewitt, magNormalizedPrewitt, 0, 255, NORM_MINMAX, CV_8UC1);

		Mat magSobel = magnitude(src, "Sobel");
		normalize(magSobel, magNormalizedSobel, 0, 255, NORM_MINMAX, CV_8UC1);

		Mat magRoberts = magnitude(src, "Roberts");
		normalize(magRoberts, magNormalizedRoberts, 0, 255, NORM_MINMAX, CV_8UC1);

		Mat dirPrewitt = direction(src, "Prewitt");
		normalize(dirPrewitt, dirNormalizedPrewitt, 0, 255, NORM_MINMAX, CV_8UC1);

		Mat dirSobel = direction(src, "Sobel");
		normalize(dirSobel, dirNormalizedSobel, 0, 255, NORM_MINMAX, CV_8UC1);

		Mat dirRoberts = direction(src, "Roberts");
		normalize(dirRoberts, dirNormalizedRoberts, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow("input image", src);

		imshow("magnitude Prewitt", magNormalizedPrewitt);
		imshow("magnitude Sobel", magNormalizedSobel);
		imshow("magnitude Roberts", magNormalizedRoberts);

		imshow("direction Prewitt", dirNormalizedPrewitt);
		imshow("direction Sobel", dirNormalizedSobel);
		imshow("direction Roberts", dirNormalizedRoberts);
		waitKey();
	}
}

Mat convulutionGradient(Mat src, Mat kernel) {

	Mat dst = Mat(src.rows, src.cols, CV_32SC1);

	int i, j, u, v;
	int k = kernel.rows / 2;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			int sum = 0;
			for (u = 0; u < kernel.rows; u++) {
				for (v = 0; v < kernel.cols; v++) {
					if (isInside(src, i + u - k, j + v - k)) {
						sum += kernel.at<int>(u, v) * src.at<uchar>(i + u - k, j + v - k);
						dst.at<int>(i, j) = sum;
					}
				}
			}
		}
	}
	return dst;
}

// functie bazata pe cea din laboratorul 8 cu contrastul
Mat normalization(Mat src, int goutMin, int goutMax) {
	int ginMin = src.at<int>(0, 0);
	int ginMax = src.at<int>(0, 0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<int>(i, j) < ginMin) {
				ginMin = src.at<int>(i, j);
			}
			if (src.at<int>(i, j) > ginMax) {
				ginMax = src.at<int>(i, j);
			}
		}
	}
	float value = (goutMax - goutMin) / (float)(ginMax - ginMin);

	Mat dst = Mat(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int val = goutMin + ((src.at<int>(i, j) - ginMin) * value);
			dst.at<uchar>(i, j) = val;
		}
	}
	return dst;
}


Mat magnitude(Mat gradient_X, Mat gradient_Y) {
	int i, j;
	Mat dst = Mat(gradient_X.rows, gradient_X.cols, CV_32SC1, cv::Scalar(0));
	for (i = 0; i < gradient_X.rows; i++) {
		for (j = 0; j < gradient_X.cols; j++) {
			dst.at<int>(i, j) = (int)sqrt((gradient_X.at<int>(i, j) * gradient_X.at<int>(i, j))
				+ (gradient_Y.at<int>(i, j) * gradient_Y.at<int>(i, j)));
		}
	}
	return dst;
}

Mat direction(Mat gradient_X, Mat gradient_Y) {
	int i, j;
	Mat dst = Mat(gradient_X.rows, gradient_X.cols, CV_32SC1, cv::Scalar(0));
	for (i = 0; i < gradient_X.rows; i++) {
		for (j = 0; j < gradient_X.cols; j++) {
			dst.at<int>(i, j) = (int)((atan2(gradient_Y.at<int>(i, j), gradient_X.at<int>(i, j)) * 180) / PI);
		}
	}
	return dst;
}

Mat nonMaximumSuppression(Mat magnitude, Mat direction) {
	Mat dst = Mat(magnitude.rows, magnitude.cols, CV_32SC1, cv::Scalar(0));
	int grad;
	int i, j;

	for (i = 0; i < direction.rows; i++) {
		for (j = 0; j < direction.cols; j++) {
			if (direction.at<int>(i, j) < 0) {
				grad = direction.at<int>(i, j) + 360;
			}
			else {
				grad = direction.at<int>(i, j);
			}

			if ((grad >= 337.5 && grad <= 360) || (grad >= 0 && grad <= 22.5) || (grad >= 157.5 && grad <= 202.5)) {
				// Neighbors to the left and right
				if (isInside(magnitude, i, j - 1) && isInside(magnitude, i, j + 1)
					&& (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i, j - 1)) && (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i, j + 1))) {
					dst.at<int>(i, j) = magnitude.at<uchar>(i, j);
				}
				else {
					dst.at<int>(i, j) = 0;
				}
			}

			if ((grad >= 67.5 && grad <= 112.5) || (grad >= 247.5 && grad <= 292.5)) {
				// Neighbors above and below
				if (isInside(magnitude, i - 1, j) && isInside(magnitude, i + 1, j)
					&& (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j)) && (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j))) {
					dst.at<int>(i, j) = magnitude.at<uchar>(i, j);
				}
				else {
					dst.at<int>(i, j) = 0;
				}
			}

			if ((grad >= 112.5 && grad <= 157.5) || (grad >= 292.5 && grad <= 337.5)) {
				// Neighbors diagonally
				if (isInside(magnitude, i - 1, j - 1) && isInside(magnitude, i + 1, j + 1)
					&& (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j - 1)) && (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j + 1))) {
					dst.at<int>(i, j) = magnitude.at<uchar>(i, j);
				}
				else {
					dst.at<int>(i, j) = 0;
				}
			}

			if ((grad >= 22.5 && grad <= 67.5) || (grad >= 202.5 && grad <= 247.5)) {
				// Neighbors diagonally
				if (isInside(magnitude, i - 1, j + 1) && isInside(magnitude, i + 1, j - 1)
					&& (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j + 1)) && (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j - 1))) {
					dst.at<int>(i, j) = magnitude.at<uchar>(i, j);
				}
				else {
					dst.at<int>(i, j) = 0;
				}
			}
		}
	}
	return dst;
}

int threshold_high(Mat src, float p) {
	int* histogram = calculateHistogramFromGivenImage(src);
	int sum = 0, threshold = 0;

	int nrNonMuchii = (1 - p) * (src.rows * src.cols - histogram[0]);

	for (int i = 1; i < 256; i++) {
		sum += histogram[i];
		if (sum > nrNonMuchii) {
			threshold = i;
			break;
		}
	}
	return threshold;
}

Mat edge_extension(Mat src, int threshold_high, float k) {

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int i, j;

	int threshold_low = k * threshold_high;

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) > threshold_high) {
				dst.at<uchar>(i, j) = 255;
			}
			else if (src.at<uchar>(i, j) > threshold_low) {
				dst.at<uchar>(i, j) = 128;
			}
		}
	}
	return dst;
}


Mat edge_binarization(Mat src) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	// vecinatate
	int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int i, j;
	Mat labels = Mat(src.rows, src.cols, CV_32SC1, cv::Scalar(0));

	for (i = 0; i < src.rows; i++) {
		for (j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {

				std::queue<Point2i> q;
				q.push(Point2i(i, j));
				dst.at<uchar>(i, j) = 255;

				while (!q.empty()) {
					Point2i p = q.front();
					q.pop();
					for (int k = 0; k < 8; k++) {
						if (isInside(src, p.x + di[k], p.y + dj[k]) && (src.at<uchar>(p.x + di[k], p.y + dj[k]) == 128) && (labels.at<int>(p.x + di[k], p.y + dj[k]) == 0)) {
							labels.at<int>(p.x + di[k], p.y + dj[k]) = 1;
							dst.at<uchar>(p.x + di[k], p.y + dj[k]) = 255;
							q.push(Point2i(p.x + di[k], p.y + dj[k]));
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat_<float> createBidimensionalGaussianKernel(int w) {
	double t = (double)getTickCount();

	Mat_<float> kernel = Mat(w, w, CV_32FC1);

	int k = w / 2;
	float sigma = w / 6.0;
	float expression;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			expression = -(((i - k) * (i - k) + (j - k) * (j - k)) / (2.0 * sigma * sigma));
			kernel(i, j) = (1.0 / (2.0 * PI * sigma * sigma)) * exp(expression);
		}
	}

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("\nTime = %.3f [ms] \n", t * 1000);

	return kernel;
}

Mat convolutionWithBidimensionalGaussianKernel(Mat img, int w) {
	Mat_<float> kernel = createBidimensionalGaussianKernel(w);
	return convolution(img, kernel);
}

Mat canny(Mat src) {
	// filtrare cu filtru Gaussian pentru eliminarea zgomotelor
	Mat img = convolutionWithBidimensionalGaussianKernel(src, 3);

	// definire kernel (Sobel)
	std::pair<Mat_<int>, Mat_<int>> sobel = defineGradientComponentsSobel("Sobel");

	Mat sobel_X = sobel.first;
	Mat sobel_Y = sobel.second;

	// determinarea gradientului
	Mat gradient_X = convulutionGradient(img, sobel_X);
	Mat gradient_Y = convulutionGradient(img, sobel_Y);

	// normalizare in intervalul [0-255]
	Mat gradient_X_norm = normalization(gradient_X, 0, 255);
	Mat gradient_Y_norm = normalization(gradient_Y, 0, 255);

	// determinare magnitudine si directie pentru gradient
	Mat magn = magnitude(gradient_X, gradient_Y);
	Mat theta = direction(gradient_X, gradient_Y);

	// normalizare in intervalul [0-255]
	Mat magnitude_norm = normalization(magn, 0, 255);
	Mat direction_norm = normalization(theta, 0, 255);

	// suprimare non-maxime
	Mat suprresion = nonMaximumSuppression(magnitude_norm, theta);

	// normalizare in intervalul [0-255]
	Mat suppresion_norm = normalization(suprresion, 0, 255);

	// determinare prag inalt
	int threshold = threshold_high(suppresion_norm, 0.1);

	// extindere muchii prin histerezis
	Mat edge_extended = edge_extension(suppresion_norm, threshold, 0.4);

	//binarizare muchii
	Mat dst = edge_binarization(edge_extended);

	return dst;
}


// Rectangle detection (using Douglas-Peucker algorithm)
void cnt_rect(const std::vector<std::vector<Point>>& cnts, std::vector<Point>& result, double coef = 0.1) {
	std::vector<std::vector<cv::Point>> contour_list;
	for (const auto& cnt : cnts) {
		double peri = arcLength(cnt, true);
		std::vector<Point> approx;
		approxPolyDP(cnt, approx, coef * peri, true);
		if (approx.size() == 4) {
			contour_list.push_back(cnt);
		}
	}
	if (!contour_list.empty()) {
		result = *std::max_element(contour_list.begin(), contour_list.end(),
			[](const std::vector<Point>& a, const std::vector<Point>& b) {
				return contourArea(a) < contourArea(b);
			});
	}
	else {
		result.clear();
	}
}

// Circle detection
void cnt_circle(Mat img, std::vector<Point>& result, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius) {
	Mat mask = Mat::zeros(img.size(), CV_8UC1);
	std::vector<Vec3f> circles;
	HoughCircles(img, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
	if (circles.empty()) {
		result.clear();
		return;
	}

	auto largest_circle = *std::max_element(circles.begin(), circles.end(),
		[](const cv::Vec3f& a, const cv::Vec3f& b) {
			return a[2] < b[2];
		});

	int center_x = cvRound(largest_circle[0]);
	int center_y = cvRound(largest_circle[1]);
	int radius = cvRound(largest_circle[2]);

	circle(mask, Point(center_x, center_y), radius, 255, -1);
	std::vector<std::vector<cv::Point>> cnts;
	findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (!cnts.empty()) {
		result = *std::max_element(cnts.begin(), cnts.end(),
			[](const std::vector<Point>& a, const std::vector<Point>& b) {
				return cv::contourArea(a) < cv::contourArea(b);
			});
	}
	else {
		result.clear();
	}
}

Mat integrate_circle_rect(cv::Mat rect_cnt, cv::Mat circle_cnt, std::vector<std::vector<cv::Point>>& cnt) {
	// Check for empty inputs first
	if (circle_cnt.empty() && rect_cnt.empty()) {
		return cv::Mat();
	}

	// Prioritize returning the larger contour (circle or rectangle)
	if (!circle_cnt.empty() && !rect_cnt.empty()) {
		return cv::contourArea(circle_cnt) >= cv::contourArea(rect_cnt) ? circle_cnt : rect_cnt;
	}

	return circle_cnt.empty() ? rect_cnt : circle_cnt;
}

Mat integrate_edge_color(Mat output1, Mat output2) {
	if (!output1.empty() && !output2.empty()) {
		// Compare contour area
		if (cv::contourArea(output1) > cv::contourArea(output2)) {
			return output1;
		}
		else {
			return output2;
		}
	}
	else if (!output1.empty() && output2.empty()) {
		return output1;
	}
	else if (output1.empty() && !output2.empty()) {
		return output2;
	}
	else {
		return cv::Mat();
	}
}

void test_shape_detection() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname);

		// Convert the image to grayscale - it can be loaded directly in grayscale
		Mat gray;
		cvtColor(src, gray, COLOR_BGR2GRAY);

		// Apply the Canny edge detector
		Mat edges = canny(gray);
		imshow("canny", edges);

		// Find contours
		std::vector<std::vector<cv::Point>> contours;
		findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// Detect rectangles and circles
		std::vector<Point> rect_result;
		cnt_rect(contours, rect_result);

		std::vector<Point> circle_result;
		cnt_circle(gray, circle_result, 1, 20, 50, 30, 0, 0);

		// Integrate the results
		Mat rect_cnt = rect_result.empty() ? Mat() : Mat(rect_result);
		Mat circle_cnt = circle_result.empty() ? Mat() : Mat(circle_result);

		Mat final_result = integrate_circle_rect(rect_cnt, circle_cnt, contours);

		if (!final_result.empty()) {
			if (final_result.channels() == 1) {
				std::vector<std::vector<Point>> single_contour = { final_result };
				drawContours(src, single_contour, -1, cv::Scalar(0, 255, 0), 2);
			}
			else {
				std::vector<Point> points;
				final_result.copyTo(points);
				std::vector<std::vector<Point>> single_contour = { points };
				drawContours(src, single_contour, -1, cv::Scalar(0, 255, 0), 2);
			}
		}

		imshow("Detected Shapes", src);
		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Color Based Detections\n");
		printf(" 11 - Canny Based Detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			signDetectoion();
			break;
		case 11:
			test_shape_detection();
			break;
		}
	} while (op != 0);
	return 0;
}