#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "queue.h"
 
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include <cstdlib>
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <arpa/inet.h>
#include <fstream>
#include <sys/time.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;
using namespace dlib;

#define KEY_ESC 27

// FACE ROTATION
float alpha; 
float devLeft = 0, devRight = 0, devFace = 0; // deviation
// VIEW PCL
float xObj2, yObj2, zObj2; 
int   xNose2D, yNose2D, xLeft2D, yLeft2D, xRight2D, yRight2D, xAbove2D, yAbove2D, xBelow2D, yBelow2D;
float xNose3D, yNose3D, zNose3D,
      xRight3D, yRight3D, zRight3D,
      xLeft3D, yLeft3D, zLeft3D,
      xAbove3D, yAbove3D, zAbove3D,
      xBelow3D, yBelow3D, zBelow3D;
// THRESHOLD POINT CLOUD
float dFront = 1,
	  dBehind = 5,
	  dLeft = 20,
	  dRight = 10,
      dHeight_Above = 200,
      dHeight_Below = 5;
// CALIB PARAMETER
Mat R1, R2, P1, P2, Q;
Mat K1, K2, R, D1, D2; 
Vec3d T;
Mat lmapx, lmapy, rmapx, rmapy;
// MAT IMAGES
Mat imgU1, imgU1_draw, insert, insertCrop, imgU2, imgU1Crop, imgU2Crop;
Mat img1, img2;
Mat pointcloud;
Mat imgDisparity16S,
	Disparity16S,
	imgDisparity8U;
// SYNCHORONIZATION
int   syn1, syn2;
int   thresItensityLeft = 30, thresItensityRight = 40 ;
int   windowSizeItensity = 30;
float BrightnessCamLeft_first; 
float BrightnessCamRight_first;
float BrightnessCamLeft_current; 
float BrightnessCamRight_current;
bool  checkFirstFrame = false;
bool  checkSyn1 = false;
bool  checkSyn2 = false;
bool  stopQueue = true;
int   countFrameIR = 0;
// TIME PROCESS
int fps = 0; 
int count_ = 0;
int countFrame = 1;
// WRITE QUEUE
QUEUE *qLeft = new QUEUE();
QUEUE *qRight = new QUEUE();
// THRESHOLD PROCESSING
int xVertexRec = 600,   // Vertex of the rectangle. 
    yVertexRec = 200,
    xoptVertexRec = 1240, // Vertex of the rectangle opposite to (xVertexRec, yVertexRec).
    yoptVertexRec = 840;
Rect rectCrop(xVertexRec, yVertexRec, xoptVertexRec - xVertexRec, yoptVertexRec - yVertexRec); // crop size
// INPUT VIDEO
struct paraReadVideo
{
	VideoCapture vid_left;
	VideoCapture vid_right;
};
// SHOW
Mat show1, show2, show2_12; 
/*********** Pattern - IR controller ***************/
char *ipKit = "10.11.11.74";
struct sockaddr_in address;
int sock = 0, valread;
struct sockaddr_in serv_addr;
char *message = new char[5];
char *buffer = new char[10];
bool onBlink = false;
#define DEFAULT_PORT 5000

/*------------- SAVE RESULT -----------------*/
char saveLeft[80];char saveRight[80];char saveLeftR[80];char saveRightR[80];char saveDis[80];char saveIns[80];char savePCL[80];
/*------------Config-----------------_*/
bool firstTime;
int countOn;

//--Facial Landmark Detection
void face_landmark(const char* shape, cv::Mat &image)
{
	try
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize(shape) >> sp;
		cv_image<bgr_pixel> img(image);
		std::vector<dlib::rectangle> dets = detector(img);
		cout << "Number of faces detected: " << dets.size() << endl;
		std::vector<full_object_detection> shapes;

		for (unsigned long j = 0; j < dets.size(); ++j)
		{
		    full_object_detection shape = sp(img, dets[j]);

		    xNose2D = shape.part(30)(0); 
		    yNose2D = shape.part(30)(1);
		    xLeft2D = shape.part(1)(0);
		    yLeft2D = shape.part(1)(1);
		    xRight2D = shape.part(15)(0);
		    yRight2D = shape.part(15)(1);
		    xAbove2D = shape.part(19)(0);
		    yAbove2D = shape.part(19)(1);
		    xBelow2D = shape.part(8)(0);
		    yBelow2D = shape.part(8)(1);

		    devLeft = sqrt((xNose2D - xLeft2D)*(xNose2D - xLeft2D)+(yNose2D - yLeft2D)*(yNose2D - yLeft2D));
		    devRight = sqrt((xRight2D - xNose2D)*(xRight2D - xNose2D)+(yRight2D - yNose2D)*(yRight2D - yNose2D));
		    devFace = devRight - devLeft;
		    // cout << "Deviation: " << devFace << endl;
		    shapes.push_back(shape);
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}

void Calculate_Position_Camera(float xObj1, float yObj1, float zObj1, float alpha)
{
	xObj2 = zObj1*sin(alpha) + xObj1;
	yObj2 = yObj1;
	zObj2 = zObj1*(1 - cos(alpha));
}

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (255, 255, 255);
    viewer.setCameraPosition(xObj2, yObj2, zObj2, xNose3D, yNose3D, zNose3D,  0, -1, 0);
}

static unsigned int var_size = 3;
void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::VScroll)
    {
      if (event.getType () == pcl::visualization::MouseEvent::MouseScrollUp)
          {
          if(var_size <=15)
          var_size ++;
          else
              var_size = 16;
              std::cout << "Scroll Down: pointsize " << var_size<<" " << std::endl;
          }
       if (event.getType () == pcl::visualization::MouseEvent::MouseScrollDown)
          {
           if (var_size >=4)
           var_size --;
           else var_size = 3;
              std::cout << "Scroll Up: pointsize" << var_size<<" " << std::endl;
          }
    }
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, var_size, "Scan cloud");
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (255, 255, 255);
	viewer->setCameraPosition(xObj2, yObj2, zObj2, xNose3D, yNose3D, zNose3D,  0, -1, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "Scan cloud");
	std::cout<< "Viewer pointSize" << var_size << endl;
	viewer->registerMouseCallback (mouseEventOccurred, (void*)viewer.get ());
	return (viewer);
}

void draw_cloud(
    const std::string &text,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    bool resetblink = false;
    pcl::visualization::CloudViewer viewer(text);
    viewer.showCloud(cloud);
    viewer.runOnVisualizationThreadOnce(viewerOneOff);
    while (!viewer.wasStopped())
    {
    	if (!resetblink){
    		sock = socket(AF_INET, SOCK_STREAM, 0);
			// Send signal to off Blink
			message = "5\0";
			connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
		  	send(sock , message , strlen(message) , 0 );
			shutdown(sock, SHUT_WR);
		  	printf("Message sent to %s: %s\n",ipKit, message);
			valread = read(sock , buffer, 10);
		
			printf("XXX: %s\n",buffer);
		  	close(sock);
			// end
			resetblink = true;
			stopQueue = false;
			while (!qLeft->isEmpty())
	    		qLeft->Pop();
	    	while (!qRight->isEmpty())
	    		qRight->Pop();
    	}

    	int key1 = waitKey(1);
    	if (key1 == 'r'){
    		checkFirstFrame = false;
    		stopQueue = true;
    		checkSyn1 = false; 
    		checkSyn2 = false;
    		countFrameIR = 0;
			countFrame = 1;
    		count_ = 0;
    		syn1 = 0;
        	syn2 = 0;
    		break;
    	}
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr img_to_cloud(
        const cv::Mat &image, cv::Mat &pointcloud, float dFront, float dBehind, float dLeft, float dRight, float dHeight_Above, float dHeight_Below)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    Vec3f point1;
    for (int y=0;y<image.rows;y++)
    {
        for (int x=0;x<image.cols;x++)
        {
            point1 = pointcloud.at<Vec3f>(y, x);

            if (x == xNose2D && y == yNose2D)
            {
				xNose3D = point1[0];
				yNose3D = point1[1];
				zNose3D = point1[2];
				cout << "Nose: " << x << " " << y << " " << xNose3D << " " << yNose3D << " " << zNose3D << endl;
            }

            if (x == xLeft2D && y == yLeft2D)
            {
				xLeft3D = point1[0];
				yLeft3D = point1[1];
				zLeft3D = point1[2];
				cout << "Left: " << x << " " << y << " " << xLeft3D << " " << yLeft3D << " " << zLeft3D << endl;
            }

            if (x == xRight2D && y == yRight2D)
            {
				xRight3D = point1[0];
				yRight3D = point1[1];
				zRight3D = point1[2];
				cout << "Right: " << x << " " << y << " " << xRight3D << " " << yRight3D << " " << zRight3D << endl;
            }
        }
    }
   
    for (int y=0;y<image.rows;y++)
    {
        for (int x=0;x<image.cols;x++)
        {
          	point1 = pointcloud.at<Vec3f>(y, x);  
            if (x >= xLeft2D - dLeft && x <= xRight2D + dRight && y <= yBelow2D + dHeight_Below && y >= yAbove2D - dHeight_Above && point1[2] >= zNose3D - dFront && point1[2] <= zNose3D + dBehind)
            {   
				pcl::PointXYZRGB point;
				point.x = point1[0];
				point.y = point1[1];
				point.z = point1[2];
				cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(x,y));
				//--RGB
				uint8_t r = (color[2]);
				uint8_t g = (color[1]);
				uint8_t b = (color[0]);

				//--BLACK
				// uint8_t r = 0;
				// uint8_t g = 0;
				// uint8_t b = 0;

				int32_t rgb = (r << 16) | (g << 8) | b;
				point.rgb = *reinterpret_cast<float*>(&rgb);

				cloud->points.push_back(point);
            }
        }
    }    
    //--StatisticalOutlierRemoval Filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(500);
	sor.setStddevMulThresh(1);
	sor.filter(*cloud_filtered);

    return cloud_filtered;
}

float itensitySizeBelow(Mat &img, int windowSize)
{
	int sum = 0;
	for (int i = img.rows; i > img.rows - windowSize; i--)
		for (int j = img.cols/2 - windowSize/2; j < img.cols/2 + windowSize/2; j++)
			sum += (float)img.at<uchar>(i, j);
	// cout << (float)sum/(windowSize*windowSize) << endl;
	return (float)sum/(windowSize*windowSize);
}

float itensitySizeCenter(Mat &img, int windowSize)
{
	int sum = 0;
	for (int i = img.rows/2 - windowSize/2; i < img.rows/2 + windowSize/2; i++)
		for (int j = img.cols/2 - windowSize/2; j < img.cols/2 + windowSize/2; j++)
			sum += (float)img.at<uchar>(i,j);
	// cout << (float)sum/(windowSize*windowSize) << endl;
	return (float)sum/(windowSize*windowSize);
}

void ProcessingSGM(Mat imgLeft, Mat imgRight)
{
	clock_t time1;
	cv::remap(imgLeft, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
	cv::remap(imgRight, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
	imgU1_draw = imgU1.clone();
	cv::rectangle( imgU1_draw, Point( xVertexRec, yVertexRec ), Point( xoptVertexRec, yoptVertexRec ), Scalar( 0, 0, 255 ), 3, 4 );
	putText(imgU1_draw, "Processing Area", cvPoint(550,100), FONT_HERSHEY_COMPLEX_SMALL, 4, cvScalar(255,255,255), 1, CV_AA);
	putText(imgU1_draw, "FPS: " + to_string(fps), cvPoint(10,100), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(255,255,255), 1, CV_AA);
	// Area process
	imgU1Crop = imgU1(rectCrop); 
	imgU2Crop = imgU2(rectCrop);

	imgDisparity16S = Mat( imgU1Crop.rows, imgU1Crop.cols, CV_16S );
	Disparity16S = Mat( imgU1.rows, imgU1.cols, CV_16S, Scalar(0) );
	imgDisparity8U = Mat( imgU1.rows, imgU1.cols, CV_8UC1 );

	Ptr<StereoSGBM> sgbm = StereoSGBM::create();
	sgbm->setMinDisparity(1);
	sgbm->setNumDisparities(16*10);
	sgbm->setBlockSize(15);
	// sgbm->setP1(0); // 8
	sgbm->setP2(512); // 32
	// sgbm->setDisp12MaxDiff(0);
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
	hconcat(imgU1_draw, imgDisparity8U, show2_12);
	hconcat(show2_12, imgU2, show2);
	imshow("INPUT PROCESS", show2);
	if (!checkFirstFrame){
		BrightnessCamLeft_first = itensitySizeBelow(imgLeft, windowSizeItensity);
		BrightnessCamRight_first = itensitySizeBelow(imgRight, windowSizeItensity);
		checkFirstFrame = true;
	}
	BrightnessCamLeft_current = (float)itensitySizeBelow(imgLeft, windowSizeItensity)/BrightnessCamLeft_first;
	BrightnessCamRight_current = (float)itensitySizeBelow(imgRight, windowSizeItensity)/BrightnessCamRight_first;
	
	cout << itensitySizeBelow(imgLeft, windowSizeItensity) << " " << itensitySizeBelow(imgRight, windowSizeItensity) << " " << BrightnessCamLeft_current<<" "<<BrightnessCamRight_current<<" "<<BrightnessCamLeft_first<< " " << BrightnessCamRight_first<< endl;
	if (BrightnessCamLeft_current > 0.7 ){
		insert = imgU1.clone();
		cvtColor(insert, insert, cv::COLOR_GRAY2BGR);
	}
	else {
		countFrameIR ++;
		if (countFrameIR == 2) {
			struct timeval tp;
			gettimeofday(&tp, NULL);
			long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

			
			
			sprintf(saveLeft,"../Output/Left/%03ld.jpg",ms);
			
			
			sprintf(saveRight,"../Output/Right/%03ld.jpg",ms);

			
			sprintf(saveLeftR,"../Output/LeftR/%03ld.jpg",ms);

			
			sprintf(saveRightR,"../Output/RightR/%03ld.jpg",ms);

			
			sprintf(saveDis,"../Output/Dis/%03ld.jpg",ms);

			
			sprintf(saveIns,"../Output/Ins/%03ld.jpg",ms);

			imwrite(saveDis, imgDisparity8U);
			imwrite(saveIns, insert);
			imwrite(saveLeftR, imgU1);
			imwrite(saveRightR, imgU2);
			imwrite(saveLeft, imgLeft);
			imwrite(saveRight, imgRight);
			
			/*
			imwrite("../Output/Disparity.png", imgDisparity8U);
			imwrite("../Output/InsertRectify.png", insert);
			imwrite("../Output/LeftRectify.png", imgU1);
			imwrite("../Output/RightRectify.png", imgU2);
			imwrite("../Output/Left.png", imgLeft);
			imwrite("../Output/Right.png", imgRight);*/
			
			
			sprintf(savePCL,"../Output/Point/%03ld.ply",ms);
			insertCrop = Mat( imgU1.rows, imgU1.cols, CV_8UC3, Scalar(0) );
			insert = insert(rectCrop);
			insert.copyTo(insertCrop.rowRange(yVertexRec, yoptVertexRec).colRange(xVertexRec, xoptVertexRec));
			face_landmark("../shape_predictor_68_face_landmarks.dat", insertCrop);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = img_to_cloud(insertCrop, pointcloud, dFront, dBehind, dLeft, dRight, dHeight_Above, dHeight_Below);
		    pcl::io::savePLYFileASCII (savePCL, *cloud);
		    alpha = devFace * -0.5 ;
			alpha = alpha * 3.14 / 180.;
			Calculate_Position_Camera(xNose3D, yNose3D, zNose3D, alpha);
		    // draw_cloud("3D Viewer", cloud);

			bool resetblink = false;
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
			viewer = rgbVis(cloud);
			while (!viewer->wasStopped ())
			{
				viewer->spinOnce (100);
				boost::this_thread::sleep (boost::posix_time::microseconds (100000));

				if (!resetblink){
					sock = socket(AF_INET, SOCK_STREAM, 0);
					// Send signal to off Blink
					message = "5\0";
					connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
						send(sock , message , strlen(message) , 0 );
					shutdown(sock, SHUT_WR);
						printf("Message sent to %s: %s\n",ipKit, message);
					valread = read(sock , buffer, 10);

					printf("XXX: %s\n",buffer);
						close(sock);
					// end
					resetblink = true;
					stopQueue = false;
					while (!qLeft->isEmpty())
						qLeft->Pop();
					while (!qRight->isEmpty())
						qRight->Pop();
				}

				int key1 = waitKey(1);
				if (key1 == 'r'){
					checkFirstFrame = false;
					stopQueue = true;
					checkSyn1 = false; 
					checkSyn2 = false;
					countFrameIR = 0;
					countFrame = 1;
					count_ = 0;
					syn1 = 0;
					syn2 = 0;
					break;
				}
				}

		}
	}
	float diff(clock()-time1);
	fps = CLOCKS_PER_SEC/ diff;
	// cout <<"Estimated fps: " << fps << endl;	
}

void *readVideo(void *paraReadVideo)
{	
	struct paraReadVideo *rV = (struct paraReadVideo*) paraReadVideo;

	while(1)
	{
		if(firstTime){
			countOn = 30;
			firstTime = false;
		}
		else
			countOn = 2;
		if (count_ == countOn){
			sock = socket(AF_INET, SOCK_STREAM, 0);
			// Send signal to on Blink
			message = "4\0";
			connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
		  	send(sock , message , strlen(message) , 0 );
			shutdown(sock, SHUT_WR);
		  	printf("Message sent to %s: %s\n",ipKit, message);
			valread = read(sock , buffer, 10);
			printf("XXX: %s\n",buffer);
		  	close(sock);
		}
		rV->vid_left >> img1;
	    rV->vid_right >> img2;
	    if (img1.empty()){
			cout << "Error1" << endl;
			break;
	    }
	    if (img2.empty()){
			cout << "Error2" << endl;
			break;
	    }
	    cvtColor(img1,img1,CV_RGB2GRAY);
		cvtColor(img2,img2,CV_RGB2GRAY);
		if (itensitySizeCenter(img1, windowSizeItensity) > thresItensityLeft && checkSyn1 == false){
			syn1 = count_;
			checkSyn1 = true;			
		}
		if (itensitySizeCenter(img2, windowSizeItensity) > thresItensityRight && checkSyn2 == false){
			syn2 = count_;
			checkSyn2 = true;
		}
		++ count_;
cout << itensitySizeCenter(img1, windowSizeItensity) << "XXX" << itensitySizeCenter(img2, windowSizeItensity) << endl;
		cout << syn1 << "---" << syn2 << endl; 
		hconcat(img1, img2, show1);
		imshow("INPUT CURRENT", show1);
		if (checkSyn1 && stopQueue)
	    	qLeft->Push(img1);
	    if (checkSyn2 && stopQueue)
	    	qRight->Push(img2);
	}
}

void *processSMG(void *arg)
{
	begin:
	while(1){
		if ( syn1 != 0 && syn2 != 0){
				if (qLeft->isEmpty() || qRight->isEmpty())
					continue;
				ProcessingSGM(qLeft->Pop(), qRight->Pop());
			}
			countFrame++;
		waitKey(1);
	}
}

void *flushVideo(void *vid){
	VideoCapture *video = (VideoCapture*) vid;
	int fps = 5;
	int maxDelay = 1000/fps - 15;
	int delay = 0;
	int frameDelayCnt = 0;
	while(frameDelayCnt <= 1){
		auto start = std::chrono::high_resolution_clock::now();
		video->grab();
		auto stop = std::chrono::high_resolution_clock::now();
		auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		delay = int_ms.count();
		cout << "Delay: " << delay << endl;
		if(delay > maxDelay)
			frameDelayCnt++;
	}
}

int main(int argc, char const *argv[])
{
	// sock = socket(AF_INET, SOCK_STREAM, 0);
	// // Send signal to off Blink
	// message = "5\0";
	// connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
	// send(sock , message , strlen(message) , 0 );
	// shutdown(sock, SHUT_WR);
	// printf("Message sent to %s: %s\n",ipKit, message);
	// valread = read(sock , buffer, 10);

	// printf("XXX: %s\n",buffer);
	// close(sock);
	
	if (!fs::is_directory("../Output") || !fs::exists("../Output")) { 
    		fs::create_directory("../Output");
    		fs::create_directory("../Output/Dis");
    		fs::create_directory("../Output/Ins");
    		fs::create_directory("../Output/Left");
    		fs::create_directory("../Output/LeftR");
    		fs::create_directory("../Output/Point");
    		fs::create_directory("../Output/Right");
    		fs::create_directory("../Output/RightR");

	}
	
	firstTime = true;
	namedWindow("INPUT CURRENT", CV_WINDOW_NORMAL);
	resizeWindow("INPUT CURRENT", 1920, 540);
	namedWindow("INPUT PROCESS", CV_WINDOW_NORMAL);
	resizeWindow("INPUT PROCESS", 1440, 270);

	cv::FileStorage fs1("../calibFile/calibStereo3D.yml", cv::FileStorage::READ);
	fs1["K1"] >> K1;
	fs1["K2"] >> K2;
	fs1["D1"] >> D1;
	fs1["D2"] >> D2;
	fs1["R"] >> R;
	fs1["T"] >> T;

	stereoRectify(K1, D1, K2, D2, Size(1920, 1080), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, Size(1920, 1080));
	cv::initUndistortRectifyMap(K1, D1, R1, P1, Size(1920, 1080), CV_32F, lmapx, lmapy);
	cv::initUndistortRectifyMap(K2, D2, R2, P2, Size(1920, 1080), CV_32F, rmapx, rmapy);

	//Set up Socket
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
		printf("\n Socket creation error \n");
		return NULL;
	}

	memset(&serv_addr, '0', sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(DEFAULT_PORT);

	// Convert IPv4 and IPv6 addresses from text to binary form
	if(inet_pton(AF_INET, ipKit, &serv_addr.sin_addr)<=0) {
	  printf("\nInvalid address/ Address not supported \n");
	  return NULL;
	}
	//end setup Socket 
	// VideoCapture vid_left("/DATA/Works/Trung/code/writeVideo/videosynNEW1/left5fps.avi");
	// VideoCapture vid_right("/DATA/Works/Trung/code/writeVideo/videosynNEW1/right5fps.avi");
 	VideoCapture vid_left("rtsp://10.11.11.179:554");
	VideoCapture vid_right("rtsp://10.11.11.67:554");
	// // Read all frame in cach
	pthread_t p1, p2;
	pthread_create(&p1, NULL, &flushVideo, &vid_left);
	pthread_create(&p2, NULL, &flushVideo, &vid_right);
	pthread_join(p1, NULL);
	pthread_join(p2, NULL);
	struct paraReadVideo pRV;
	pRV.vid_left = vid_left;
	pRV.vid_right = vid_right;

	pthread_t t1, t2;
	pthread_create(&t1, NULL, &readVideo, &pRV);
	pthread_create(&t2, NULL, &processSMG, NULL);
	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
	return 0;
}
