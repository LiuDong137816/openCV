#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;
using namespace xfeatures2d;
// To create histograms of gray-level images
class Histogram1D { //将算法封装进类

private:

	int histSize[1];         // number of bins in histogram直方图中箱子(bin)个数,[1]表示只有1维
	float hranges[2];    // range of values值范围,min和max共2个值，因此定义2维浮点数组
	const float* ranges[1];  // pointer to the different value ranges值范围的指针
	int channels[1];         // channel number to be examined要检查的通道数量

public:

	Histogram1D() {

		// Prepare default arguments for 1D histogram
		histSize[0] = 256;   // 256 bins,只有1维，因此通过[0]来设置该维的箱子数
		hranges[0] = 0.0;    // from 0 (inclusive)直方图取值范围的min
		hranges[1] = 256.0;  // to 256 (exclusive)直方图取值范围的max
		ranges[0] = hranges;
		channels[0] = 0;     // we look at channel 0，1维直方图暂时只看0通道
	}

	// Sets the channel on which histogram will be calculated.
	// By default it is channel 0.设置通道的方法
	void setChannel(int c) {

		channels[0] = c;
	}

	// Gets the channel used.获取通道的方法
	int getChannel() {

		return channels[0];
	}

	// Sets the range for the pixel values.设置直方图值的范围
	// By default it is [0,256]
	void setRange(float minValue, float maxValue) {

		hranges[0] = minValue;
		hranges[1] = maxValue;
	}

	// Gets the min pixel value.
	float getMinValue() {

		return hranges[0];
	}

	// Gets the max pixel value.
	float getMaxValue() {

		return hranges[1];
	}

	// Sets the number of bins in histogram.设置直方图箱子数（统计多少个灰度级）
	// By default it is 256.构造函数中默认设置为256
	void setNBins(int nbins) {

		histSize[0] = nbins;
	}

	// Gets the number of bins in histogram.
	int getNBins() {

		return histSize[0];
	}

	// Computes the 1D histogram.自编函数计算1维直方图
	cv::Mat getHistogram(const cv::Mat &image) {//输入图像image

		cv::Mat hist;

		// Compute histogram
		cv::calcHist(&image,
			1,		// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used，不使用掩码
			hist,		// the resulting histogram
			1,		// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the 1D histogram and returns an image of it.直方图图像
	cv::Mat getHistogramImage(const cv::Mat &image, int zoom = 1) {

		// Compute histogram first
		cv::Mat hist = getHistogram(image);

		// Creates image
		return Histogram1D::getImageOfHistogram(hist, zoom);
	}

	// Stretches the source image using min number of count in bins.
	cv::Mat stretch(const cv::Mat &image, int minValue = 0) {

		// Compute histogram first
		cv::Mat hist = getHistogram(image);

		// find left extremity of the histogram
		int imin = 0;
		for (; imin < histSize[0]; imin++) {
			// ignore bins with less than minValue entries
			if (hist.at<float>(imin) > minValue)
				break;
		}

		// find right extremity of the histogram
		int imax = histSize[0] - 1;
		for (; imax >= 0; imax--) {

			// ignore bins with less than minValue entries
			if (hist.at<float>(imax) > minValue)
				break;
		}

		// Create lookup table
		int dims[1] = { 256 };
		cv::Mat lookup(1, dims, CV_8U);

		for (int i = 0; i < 256; i++) {

			if (i < imin) lookup.at<uchar>(i) = 0;
			else if (i > imax) lookup.at<uchar>(i) = 255;
			else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
		}

		// Apply lookup table
		cv::Mat result;
		result = applyLookUp(image, lookup);

		return result;
	}

	// Stretches the source image using percentile.
	cv::Mat stretch(const cv::Mat &image, float percentile) {

		// number of pixels in percentile
		float number = image.total()*percentile;

		// Compute histogram first
		cv::Mat hist = getHistogram(image);

		// find left extremity of the histogram
		int imin = 0;
		for (float count = 0.0; imin < histSize[0]; imin++) {
			// number of pixel at imin and below must be > number
			if ((count += hist.at<float>(imin)) >= number)
				break;
		}

		// find right extremity of the histogram
		int imax = histSize[0] - 1;
		for (float count = 0.0; imax >= 0; imax--) {
			// number of pixel at imax and below must be > number
			if ((count += hist.at<float>(imax)) >= number)
				break;
		}

		// Create lookup table
		int dims[1] = { 256 };
		cv::Mat lookup(1, dims, CV_8U);

		for (int i = 0; i < 256; i++) {

			if (i < imin) lookup.at<uchar>(i) = 0;
			else if (i > imax) lookup.at<uchar>(i) = 255;
			else lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
		}

		// Apply lookup table
		cv::Mat result;
		result = applyLookUp(image, lookup);

		return result;
	}

	// static methods

	// Create an image representing a histogram
	static cv::Mat getImageOfHistogram(const cv::Mat &hist, int zoom) {

		// Get min and max bin values
		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		// get histogram size
		int histSize = hist.rows;

		// Square image on which to display histogram
		cv::Mat histImg(histSize*zoom, histSize*zoom, CV_8U, cv::Scalar(255));

		// set highest point at 90% of nbins (i.e. image height)
		int hpt = static_cast<int>(0.9*histSize);

		// Draw vertical line for each bin
		for (int h = 0; h < histSize; h++) {

			float binVal = hist.at<float>(h);
			if (binVal > 0) {
				int intensity = static_cast<int>(binVal*hpt / maxVal);
				cv::line(histImg, cv::Point(h*zoom, histSize*zoom),
					cv::Point(h*zoom, (histSize - intensity)*zoom), cv::Scalar(0), zoom);
			}
		}

		return histImg;
	}

	// Equalizes the source image.
	static cv::Mat equalize(const cv::Mat &image) {

		cv::Mat result;
		cv::equalizeHist(image, result);

		return result;
	}

	// Applies a lookup table transforming an input image into a 1-channel image
	static cv::Mat applyLookUp(const cv::Mat& image, // input image
		const cv::Mat& lookup) { // 1x256 uchar matrix

		// the output image
		cv::Mat result;

		// apply lookup table
		cv::LUT(image, lookup, result);

		return result;
	}
};

class ColorHistogram {

private:

	int histSize[3];         // size of each dimension
	float hranges[2];    // range of values
	const float* ranges[3];  // array of ranges for each dimension
	int channels[3];         // channel to be considered

public:

	ColorHistogram() {

		// Prepare default arguments for a color histogram
		// each dimension has equal size and range
		histSize[0] = histSize[1] = histSize[2] = 256;
		hranges[0] = 0.0;    // BRG range from 0 to 256
		hranges[1] = 256.0;
		ranges[0] = hranges; // in this class,  
		ranges[1] = hranges; // all channels have the same range
		ranges[2] = hranges;
		channels[0] = 0;	    // the three channels 
		channels[1] = 1;
		channels[2] = 2;
	}

	// set histogram size for each dimension
	void setSize(int size) {

		// each dimension has equal size 
		histSize[0] = histSize[1] = histSize[2] = size;
	}

	// Computes the histogram.
	cv::Mat getHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// BGR color histogram
		hranges[0] = 0.0;    // BRG range
		hranges[1] = 256.0;
		channels[0] = 0;		// the three channels 
		channels[1] = 1;
		channels[2] = 2;

		// Compute histogram
		cv::calcHist(&image,
			1,		// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,		// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);
		return hist;
	}
	// Computes the histogram.
	cv::SparseMat getSparseHistogram(const cv::Mat &image) {

		cv::SparseMat hist(3,        // number of dimensions
			histSize, // size of each dimension
			CV_32F);

		// BGR color histogram
		hranges[0] = 0.0;    // BRG range
		hranges[1] = 256.0;
		channels[0] = 0;	    // the three channels 
		channels[1] = 1;
		channels[2] = 2;

		// Compute histogram
		cv::calcHist(&image,
			1,		// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,		// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);
		return hist;
	}

	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	// Pixels with low saturation are ignored
	cv::Mat getHueHistogram(const cv::Mat &image,
		int minSaturation = 0) {
		cv::Mat hist;

		// Convert to HSV colour space
		cv::Mat hsv;
		cv::cvtColor(image, hsv, CV_BGR2HSV);

		// Mask to be used (or not)
		cv::Mat mask;

		if (minSaturation > 0) {
			// Spliting the 3 channels into 3 images
			std::vector<cv::Mat> v;
			cv::split(hsv, v);

			// Mask out the low saturated pixels
			cv::threshold(v[1], mask, minSaturation, 255,
				cv::THRESH_BINARY);
		}

		// Prepare arguments for a 1D hue histogram
		hranges[0] = 0.0;    // range is from 0 to 180
		hranges[1] = 180.0;
		channels[0] = 0;    // the hue channel 

		// Compute histogram
		cv::calcHist(&hsv,
			1,		// histogram of 1 image only
			channels,	// the channel used
			mask,		// binary mask
			hist,		// the resulting histogram
			1,		// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);
		return hist;
	}

	// Computes the 2D ab histogram.
	// BGR source image is converted to Lab
	cv::Mat getabHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// Convert to Lab color space
		cv::Mat lab;
		cv::cvtColor(image, lab, CV_BGR2Lab);

		// Prepare arguments for a 2D color histogram
		hranges[0] = 0;
		hranges[1] = 256.0;
		channels[0] = 1; // the two channels used are ab 
		channels[1] = 2;

		// Compute histogram
		cv::calcHist(&lab,
			1,		    // histogram of 1 image only
			channels,	    // the channel used
			cv::Mat(),	    // no mask is used
			hist,		    // the resulting histogram
			2,		    // it is a 2D histogram
			histSize,	    // number of bins
			ranges		    // pixel value range
		);
		return hist;
	}

	Mat colorReduce(const Mat &image, int div = 64)
	{
		int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));
		uchar mask = 0xFF << n;
		Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
		Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();
		//设置输出图像
		Mat result(image.rows, image.cols, image.type());
		Mat_<Vec3b>::iterator itr = result.begin<Vec3b>();
		for (; it != itend; ++it, ++itr)
		{
			(*itr)[0] = ((*it)[0] & mask) + div / 2;
			(*itr)[1] = ((*it)[1] & mask) + div / 2;
			(*itr)[2] = ((*it)[2] & mask) + div / 2;
		}
		return result;
	}
};

class ContentFinder {

private:

	// histogram parameters
	float hranges[2];
	const float* ranges[3];
	int channels[3];

	float threshold;           // decision threshold
	cv::Mat histogram;         // histogram can be sparse 输入直方图
	cv::SparseMat shistogram;  // or not
	bool isSparse;

public:

	ContentFinder() : threshold(0.1f), isSparse(false) {

		// in this class,
		// all channels have the same range
		ranges[0] = hranges;
		ranges[1] = hranges;
		ranges[2] = hranges;
	}

	// Sets the threshold on histogram values [0,1]
	void setThreshold(float t) {

		threshold = t;
	}

	// Gets the threshold
	float getThreshold() {

		return threshold;
	}

	// Sets the reference histogram
	void setHistogram(const cv::Mat& h) {

		isSparse = false;
		cv::normalize(h, histogram, 1.0);
	}

	// Sets the reference histogram
	void setHistogram(const cv::SparseMat& h) {

		isSparse = true;
		cv::normalize(h, shistogram, 1.0, cv::NORM_L2);
	}

	// All channels used, with range [0,256]
	cv::Mat find(const cv::Mat& image) {

		cv::Mat result;

		hranges[0] = 0.0;	// default range [0,256]
		hranges[1] = 256.0;
		channels[0] = 0;		// the three channels 
		channels[1] = 1;
		channels[2] = 2;

		return find(image, hranges[0], hranges[1], channels);
	}

	// Finds the pixels belonging to the histogram
	cv::Mat find(const cv::Mat& image, float minValue, float maxValue, int *channels) {

		cv::Mat result;

		hranges[0] = minValue;
		hranges[1] = maxValue;

		if (isSparse) { // call the right function based on histogram type

			for (int i = 0; i < shistogram.dims(); i++)
				this->channels[i] = channels[i];

			cv::calcBackProject(&image,
				1,            // we only use one image at a time
				channels,     // vector specifying what histogram dimensions belong to what image channels
				shistogram,   // the histogram we are using
				result,       // the resulting back projection image
				ranges,       // the range of values, for each dimension
				255.0         // the scaling factor is chosen such that a histogram value of 1 maps to 255
			);

		}
		else {

			for (int i = 0; i < histogram.dims; i++)
				this->channels[i] = channels[i];
			//某对象的this指针，指向被调用函数所在的对象，此处对象为ContentFinder类
					   //this->channels[i]即ContentFinder类的私有成员channels[3]
					   //对ContentFinder类各成员的访问均通过this进行
			cv::calcBackProject(&image,
				1,            // we only use one image at a time
				channels,     // 向量表示哪个直方图维度属于哪个图像通道
				histogram,    // 用到的直方图
				result,       // 反向投影的图像
				ranges,       // 每个维度值的范围
				255.0         // 选用的换算系数
			);
		}
		// Threshold back projection to obtain a binary image阈值分割反向投影图像得到二值图
		if (threshold > 0.0)// 设置的阈值>0时，才进行阈值分割
			cv::threshold(result, result, 255.0*threshold, 255.0, cv::THRESH_BINARY);
		return result;
	}
};

class ImageComparator
{
private:
	Mat reference;
	Mat input;
	Mat refH;
	Mat inputH;
	ColorHistogram hist;
	int div;
public:
	ImageComparator() :div(32) {}

	void setColorReducation(int factor)
	{
		div = factor;
	}

	int getColorReduction()
	{
		return div;
	}

	void setRefrenceImage(const Mat &image)
	{
		reference = hist.colorReduce(image, div);
		refH = hist.getHistogram(reference);
	}

	double compare(const Mat &image)
	{
		input = hist.colorReduce(image, div);
		inputH = hist.getHistogram(input);
		return compareHist(refH, inputH, CV_COMP_INTERSECT);
	}
};

class LineFinder {

private:
	Mat img;	//原图
	vector<Vec4i>lines;	//向量中检测到的直线的端点
	//累加器的分辨率
	double deltaRho;
	double deltaTheta;
	int minVote;	//直线被接受时所需的最小投票数
	double minLength;	//直线的最小长度
	double maxGap;	//沿着直线方向的最大缺口
public:
	//默认的累加器的分辨率为单个像素即1  不设置缺口及最小长度的值
	LineFinder() :deltaRho(1), deltaTheta(CV_PI / 180), minVote(10), minLength(0.), maxGap(0.) {};

	//设置累加器的分辨率
	void setAccResolution(double dRho, double dTheta) {

		deltaRho = dRho;
		deltaTheta = dTheta;
	}

	//设置最小投票数
	void setMinVote(int minv) {
		minVote = minv;
	}

	//设置缺口及最小长度
	void setLineLengthAndGap(double length, double gap) {
		minLength = length;
		maxGap = gap;
	}

	//使用概率霍夫变换
	vector<Vec4i>findLines(Mat &binary) {

		lines.clear();
		HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
		return lines;
	}

	//绘制检测到的直线
	void drawDetectedLines(Mat &image, Scalar color = Scalar(255, 255, 255)) {
		//画线
		vector<Vec4i>::const_iterator it2 = lines.begin();
		while (it2 != lines.end())
		{
			Point pt1((*it2)[0], (*it2)[1]);
			Point pt2((*it2)[2], (*it2)[3]);
			line(image, pt1, pt2, color);
			++it2;
		}
	}
};

class CameraCalibrator {

	//输入点
	std::vector<std::vector<cv::Point3f>> objectPoints;//世界坐标系下的点
	std::vector<std::vector<cv::Point2f>> imagePoints;//像素坐标系下的点
	//输出矩阵
	cv::Mat cameraMatrix;//摄像机内参数矩阵
	cv::Mat distCoeffs;//透镜畸变系数矩阵
	//标定方式
	int flag;
	//用于图像去畸变 
	cv::Mat map1, map2;
	bool mustInitUndistort;

public:
	CameraCalibrator() : flag(0), mustInitUndistort(true) {};

	//导入标定图片提取角点
	int addChessboardPoints(const std::vector<std::string>& filelist, cv::Size & boardSiz, std::string windowName);
	//添加场景点与对应的图像点
	void addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners);
	//标定相机
	double calibrate(const cv::Size imageSize);
	//设置标定方式
	void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);
	//消除透镜畸变(标定之后调用有效)
	cv::Mat remap(const cv::Mat &image, cv::Size &outputSize);

	// 获取矩阵
	cv::Mat getCameraMatrix() { return cameraMatrix; }
	cv::Mat getDistCoeffs() { return distCoeffs; }
};

int CameraCalibrator::addChessboardPoints(
	const std::vector<std::string>& filelist, // 文件名列表
	cv::Size & boardSize,                     // 标定面板的大小 
	std::string windowName) {                 // name of window to display results
											  // if null, no display shown
// 棋盘上的角点 
	std::vector<cv::Point2f> imageCorners;
	std::vector<cv::Point3f> objectCorners;
	// 场景中的三维点： 
	// 在棋盘坐标系中，初始化棋盘中的角点 
	// 角点的三维坐标(X,Y,Z)= (i,j,0) 
	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++) {
			objectCorners.push_back(cv::Point3f(i, j, 0.0f));
		}
	}
	// 图像上的二维点：
	cv::Mat image; // 用于存储棋盘图像 
	int successes = 0;
	// 处理所有视角 
	for (int i = 0; i < filelist.size(); i++) {
		// 打开图像
		image = cv::imread(filelist[i], 0);
		// 取得棋盘中的角点
		bool found = cv::findChessboardCorners(image,         // 包含棋盘图案的图像
			boardSize,     // 图案的大小
			imageCorners); // 检测到角点的列表
// 取得角点上的亚像素级精度
		if (found) {
			cv::cornerSubPix(image, imageCorners,
				cv::Size(5, 5), // 搜索窗口的半径 
				cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::MAX_ITER +
					cv::TermCriteria::EPS,
					30,        // 最大迭代次数 
					0.1));  // 最小精度 
			// 如果棋盘是完好的，就把它加入结果  
			if (imageCorners.size() == boardSize.area()) {
				// 加入从同一个视角得到的图像和场景点 
				addPoints(imageCorners, objectCorners);
				successes++;
			}
		}
		if (windowName.length() > 0 && imageCorners.size() == boardSize.area()) {

			//Draw the corners
			cv::drawChessboardCorners(image, boardSize, imageCorners, found);
			cv::imshow(windowName, image);
			cv::waitKey(100);
		}
	}
	return successes;
}
// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f>& imageCorners, const std::vector<cv::Point3f>& objectCorners) {
	// 2D image points from one view
	imagePoints.push_back(imageCorners);
	// corresponding 3D scene points
	objectPoints.push_back(objectCorners);
}
// 标定相机 
// 返回重投影误差 
double CameraCalibrator::calibrate(const cv::Size imageSize)
{
	mustInitUndistort = true;
	// 输出旋转量和平移量 
	std::vector<cv::Mat> rvecs, tvecs;
	// 开始标定
	return
		calibrateCamera(objectPoints, // 三维点 
			imagePoints,   // 图像点 
			imageSize,     // 图像尺寸 
			cameraMatrix,  // 输出相机矩阵 
			distCoeffs,    // 输出畸变矩阵 
			rvecs, tvecs,  // Rs、Ts 
			flag);         // 设置选项 
//                    ,CV_CALIB_USE_INTRINSIC_GUESS);
}
// 去除图像中的畸变（标定后） 
cv::Mat CameraCalibrator::remap(const cv::Mat &image, cv::Size &outputSize) {
	cv::Mat undistorted;
	if (outputSize.height == -1)
		outputSize = image.size();
	if (mustInitUndistort) { // 每个标定过程调用一次  

		cv::initUndistortRectifyMap(
			cameraMatrix,  // 计算得到的相机矩阵 
			distCoeffs,    // 计算得到的畸变矩阵 
			cv::Mat(),     // 可选矫正项（无） 
			cv::Mat(),     // 生成无畸变的相机矩阵
			outputSize,    // 无畸变图像的尺寸
			CV_32FC1,      // 输出图片的类型 
			map1, map2);   // x 和 y 映射功能 
		mustInitUndistort = false;
	}
	// Apply mapping functions
	cv::remap(image, undistorted, map1, map2,
		cv::INTER_LINEAR); // interpolation type
	return undistorted;
}
// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled) {
	// Set the flag used in cv::calibrateCamera()
	flag = 0;
	if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
	if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}

int main() {
	
	waitKey(0);
	return 0;
}