#include "opencv2/opencv.hpp"


#include <iostream>
#include <string>
#include <sys/time.h>

using namespace cv;
using namespace std;
using namespace dnn;

#define MICRO_IN_SEC 1000000.00
double microtime();
int ssd();
int main(int argc, char** argv)
{
	ssd();
	return 0;
}
double microtime() {

	struct timeval tv;
	struct timezone tz;
	gettimeofday(&tv, &tz);
	return tv.tv_sec + tv.tv_usec / MICRO_IN_SEC;
}
int ssd()
{
	double start_time, dt, dt_err;
	start_time = microtime();
	dt_err = microtime() - start_time;

	String prototxt = "MobileNetSSD_deploy.prototxt";
	String caffemodel = "MobileNetSSD_deploy.caffemodel";
	Net net = readNetFromCaffe(prototxt, caffemodel);

	const char* classNames[] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	float detect_thresh = 0.25;
	if (true)
	{
		net.setPreferableTarget(0);
	}

	VideoCapture capture(1);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	Mat image;
	int picN = 0;
	char filename[256];
	while (1)
	{
		start_time = microtime();
		int trigger = 0;
		capture >> image;
		net.setInput(blobFromImage(image, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false));
		Mat cvOut = net.forward();
		Mat detectionMat(cvOut.size[2], cvOut.size[3], CV_32F, cvOut.ptr<float>());
		for (int i = 0; i < detectionMat.rows; i++)
		{
			int obj_class = detectionMat.at<float>(i, 1);
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > detect_thresh)
			{
				size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

				ostringstream ss;
				int tmpI = 100 * confidence;
				ss << tmpI;
				String conf(ss.str());

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));
				if (classNames[objectClass] == "cat" || classNames[objectClass] == "dog")
				{
					trigger++;
					rectangle(image, object, Scalar(0, 0, 255), 1);
					String label = String(classNames[objectClass]) + ": " + conf + "%";
					putText(image, label, Point(xLeftBottom, yLeftBottom + 30 * (i + 1)), 2, 0.8, Scalar(0, 0, 255), 2);
				}
				else if (classNames[objectClass] == "pottedplant" || classNames[objectClass] == "sofa")
				{
					rectangle(image, object, Scalar(0, 255, 0), 1);
					String label = String(classNames[objectClass]) + ": " + conf + "%";
					putText(image, label, Point(xLeftBottom, yLeftBottom + 30 * (i + 1)), 2, 0.7, Scalar(0, 255, 0), 2);
				}
				else if (classNames[objectClass] == "person" || classNames[objectClass] == "bicycle")
				{
					rectangle(image, object, Scalar(255, 255, 0), 1);
					String label = String(classNames[objectClass]) + ": " + conf + "%";
					putText(image, label, Point(xLeftBottom, yLeftBottom + 30 * (i + 1)), 2, 0.7, Scalar(255, 255, 0), 2);
				}
				else
				{
					rectangle(image, object, Scalar(255, 0, 0), 1);
					String label = String(classNames[objectClass]) + ": " + conf + "%";
					putText(image, label, Point(xLeftBottom, yLeftBottom + 30 * (i + 1)), 2, 0.7, Scalar(255, 0, 0), 2);
				}

			}
		}
		if (trigger>=1)
		{
			sprintf(filename, "/home/test/dnn/detec/%06d.jpg",picN);
			picN++;
			if (picN > 10000)
				picN = 0;
			imwrite(filename, image);
			cout << "Waning" << endl;
		}
		dt = microtime() - start_time - dt_err;
		cout << "Cost time: " << dt << " s" << endl;
		imshow("test", image);
		//cv::waitKey(0);
		waitKey(1);
	}
	return 0;
}
