#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>
#include <math.h>

#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;

// THRESHOLD PROCESSING
int xVertexRec = 640,   // Vertex of the rectangle. 
    yVertexRec = 220,
    xoptVertexRec = 1280, // Vertex of the rectangle opposite to (xVertexRec, yVertexRec).
    yoptVertexRec = 860;
Rect rectCrop(xVertexRec, yVertexRec, xoptVertexRec - xVertexRec, yoptVertexRec - yVertexRec); // crop size

// CALIB PARAMETER
Mat R1, R2, P1, P2, Q;
Mat K1, K2, R, D1, D2; 
Vec3d T;
Mat lmapx, lmapy, rmapx, rmapy;

// MAT IMAGES
Mat img1, img2, 
    imgU1, imgU1_draw, imgU2,
    imgU1Crop, imgU2Crop,
    pointcloud;
Mat output2, output3;
void 
viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (255, 255, 255);
    viewer.setCameraPosition(0, 0 ,0, 0, 0, 1, 0, -1, 0);
}


void draw_cloud(
    const std::string &text,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    pcl::visualization::CloudViewer viewer(text);
    viewer.showCloud(cloud);
    viewer.runOnVisualizationThreadOnce(viewerOneOff);
    while (!viewer.wasStopped())
    { 
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr img_to_cloud(
        const cv::Mat &image, cv::Mat &pointcloud, int threshold_left, int threshold_right, int threshold_under, int threshold_above, int threshold_depth, int threshold_front)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int y=0;y<image.rows;y++)
    {
        for (int x=0;x<image.cols;x++)
        {
            Vec3f point1 = pointcloud.at<Vec3f>(y, x);
            if ( x > threshold_left && x < threshold_right && y < threshold_under && y > threshold_above && point1[2] < threshold_depth && point1[2] > threshold_front)
            {   
              pcl::PointXYZRGB point;
              point.x = point1[0];
              point.y = point1[1];
              point.z = point1[2];
              cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(x,y));
              // -----RGB
              uint8_t r = (color[2]);
              uint8_t g = (color[1]);
              uint8_t b = (color[0]);
              // -----Black
              //uint8_t r = 255;
              //uint8_t g = 255;
              //uint8_t b = 255;
              int32_t rgb = (r << 16) | (g << 8) | b;
              point.rgb = *reinterpret_cast<float*>(&rgb);
              cloud->points.push_back(point);
            }
        }
    }
    return cloud;
}


int main(int argc, char const *argv[])
{
	VideoCapture vid_left("rtsp://192.168.0.118:554");
	VideoCapture vid_right("rtsp://192.168.0.119:554");
	// // Read all frame in cach

	vid_left.release();
	vid_right.release();
	vid_left = VideoCapture("rtsp://192.168.0.118:554");
	vid_right = VideoCapture("rtsp://192.168.0.119:554");
	// VideoCapture vid_left("/DATA/Works/Trung/code/writeVideo/videosynNEW1/left5fps.avi");
	// VideoCapture vid_right("/DATA/Works/Trung/code/writeVideo/videosynNEW1/right5fps.avi");
	namedWindow( "Camera Stereo", CV_WINDOW_NORMAL);
	resizeWindow("Camera Stereo", 1920,360);

	cv::FileStorage fs1("/home/camera3d/WORKS/SynCamStereo/calibFile/cam_stereoIR_1403.yml", cv::FileStorage::READ);
	fs1["K1"] >> K1;
	fs1["K2"] >> K2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R"] >> R;
	fs1["T"] >> T;

	stereoRectify(K1, D1, K2, D2, Size(1920, 1080), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, Size(1920, 1080));
	initUndistortRectifyMap(K1, D1, R1, P1, Size(1920, 1080), CV_32F, lmapx, lmapy);
	initUndistortRectifyMap(K2, D2, R2, P2, Size(1920, 1080), CV_32F, rmapx, rmapy);
	// SET TIME
	clock_t time;
	int fps = 0;
	while(1)
	{
	time = clock();  
	vid_left >> img1;
	vid_right >> img2;
	if (img1.empty()){
	  cout << "Error1" << endl;
	  break;
	}
	if (img2.empty()){
	  cout << "Error2" << endl;
	  break;
	}

	cvtColor(img1, img1, CV_RGB2GRAY);
	cvtColor(img2, img2, CV_RGB2GRAY);
	remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
	remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
	imgU1_draw = imgU1.clone();
	cv::rectangle( imgU1_draw, Point( xVertexRec, yVertexRec ), Point( xoptVertexRec, yoptVertexRec ), Scalar( 0, 0, 255 ), 3, 4 );
	putText(imgU1_draw, "Processing Area", cvPoint(550,100), FONT_HERSHEY_COMPLEX_SMALL, 4, cvScalar(255,255,255), 1, CV_AA);
	putText(imgU1_draw, "FPS: " + to_string(fps), cvPoint(10,100), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(255,255,255), 1, CV_AA);
	// Area process
	imgU1Crop = imgU1(rectCrop); 
	imgU2Crop = imgU2(rectCrop);

	Mat imgDisparity16S = Mat( imgU1Crop.rows, imgU1Crop.cols, CV_16S );
	Mat Disparity16S = Mat( imgU1.rows, imgU1.cols, CV_16S, Scalar(0) );
	Mat imgDisparity8U = Mat( imgU1.rows, imgU1.cols, CV_8UC1 );

	Ptr<StereoSGBM> sgbm = StereoSGBM::create();
	sgbm->setMinDisparity(1);
	sgbm->setNumDisparities(16*10);
	sgbm->setBlockSize(15);
	sgbm->setP1(0); // 8
	sgbm->setP2(512); // 32
	sgbm->setDisp12MaxDiff(0);
	// sgbm->setPreFilterCap(63);
	// sgbm->setUniquenessRatio(15);
	// sgbm->setSpeckleWindowSize(100);
	// sgbm->setSpeckleRange(32);

	sgbm->compute( imgU1Crop, imgU2Crop, imgDisparity16S );
	double minVal; double maxVal;
	minMaxLoc( imgDisparity16S, &minVal, &maxVal );
	// printf("Min disp: %f Max value: %f \n", minVal, maxVal);
	imgDisparity16S.copyTo(Disparity16S.rowRange(yVertexRec, yoptVertexRec).colRange(xVertexRec, xoptVertexRec));
	Disparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
	reprojectImageTo3D(Disparity16S, pointcloud, Q, true);
	hconcat(imgU1_draw, imgU2, output2);
	hconcat(imgDisparity8U, output2, output3);
	imshow("Camera Stereo", output3);
	float diff(clock()-time);
	fps = CLOCKS_PER_SEC / diff;

	int key = waitKey(1);
	if (key == 32){
	  cvtColor(imgU1, imgU1, CV_GRAY2RGB);
	  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = img_to_cloud(imgU1, pointcloud, xVertexRec, xoptVertexRec, yoptVertexRec, yVertexRec, 65, 0);
	  imwrite("../Output/Disparity.png", imgDisparity8U);
	  imwrite("../Output/LeftRectify.png", imgU1);
	  imwrite("../Output/RightRectify.png", imgU2);
	  imwrite("../Output/Left.png", img1);
	  imwrite("../Output/Right.png", img2);
	  pcl::io::savePLYFileASCII ("../Output/3DFace.ply", *cloud);
	  draw_cloud("3D Viewer", cloud);
	}
	}
}