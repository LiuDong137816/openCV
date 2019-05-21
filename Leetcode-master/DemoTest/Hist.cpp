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
class Histogram1D { //���㷨��װ����

private:

	int histSize[1];         // number of bins in histogramֱ��ͼ������(bin)����,[1]��ʾֻ��1ά
	float hranges[2];    // range of valuesֵ��Χ,min��max��2��ֵ����˶���2ά��������
	const float* ranges[1];  // pointer to the different value rangesֵ��Χ��ָ��
	int channels[1];         // channel number to be examinedҪ����ͨ������

public:

	Histogram1D() {

		// Prepare default arguments for 1D histogram
		histSize[0] = 256;   // 256 bins,ֻ��1ά�����ͨ��[0]�����ø�ά��������
		hranges[0] = 0.0;    // from 0 (inclusive)ֱ��ͼȡֵ��Χ��min
		hranges[1] = 256.0;  // to 256 (exclusive)ֱ��ͼȡֵ��Χ��max
		ranges[0] = hranges;
		channels[0] = 0;     // we look at channel 0��1άֱ��ͼ��ʱֻ��0ͨ��
	}

	// Sets the channel on which histogram will be calculated.
	// By default it is channel 0.����ͨ���ķ���
	void setChannel(int c) {

		channels[0] = c;
	}

	// Gets the channel used.��ȡͨ���ķ���
	int getChannel() {

		return channels[0];
	}

	// Sets the range for the pixel values.����ֱ��ͼֵ�ķ�Χ
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

	// Sets the number of bins in histogram.����ֱ��ͼ��������ͳ�ƶ��ٸ��Ҷȼ���
	// By default it is 256.���캯����Ĭ������Ϊ256
	void setNBins(int nbins) {

		histSize[0] = nbins;
	}

	// Gets the number of bins in histogram.
	int getNBins() {

		return histSize[0];
	}

	// Computes the 1D histogram.�Աຯ������1άֱ��ͼ
	cv::Mat getHistogram(const cv::Mat &image) {//����ͼ��image

		cv::Mat hist;

		// Compute histogram
		cv::calcHist(&image,
			1,		// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used����ʹ������
			hist,		// the resulting histogram
			1,		// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the 1D histogram and returns an image of it.ֱ��ͼͼ��
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
		//�������ͼ��
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
	cv::Mat histogram;         // histogram can be sparse ����ֱ��ͼ
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
			//ĳ�����thisָ�룬ָ�򱻵��ú������ڵĶ��󣬴˴�����ΪContentFinder��
					   //this->channels[i]��ContentFinder���˽�г�Աchannels[3]
					   //��ContentFinder�����Ա�ķ��ʾ�ͨ��this����
			cv::calcBackProject(&image,
				1,            // we only use one image at a time
				channels,     // ������ʾ�ĸ�ֱ��ͼά�������ĸ�ͼ��ͨ��
				histogram,    // �õ���ֱ��ͼ
				result,       // ����ͶӰ��ͼ��
				ranges,       // ÿ��ά��ֵ�ķ�Χ
				255.0         // ѡ�õĻ���ϵ��
			);
		}
		// Threshold back projection to obtain a binary image��ֵ�ָ��ͶӰͼ��õ���ֵͼ
		if (threshold > 0.0)// ���õ���ֵ>0ʱ���Ž�����ֵ�ָ�
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
	Mat img;	//ԭͼ
	vector<Vec4i>lines;	//�����м�⵽��ֱ�ߵĶ˵�
	//�ۼ����ķֱ���
	double deltaRho;
	double deltaTheta;
	int minVote;	//ֱ�߱�����ʱ�������СͶƱ��
	double minLength;	//ֱ�ߵ���С����
	double maxGap;	//����ֱ�߷�������ȱ��
public:
	//Ĭ�ϵ��ۼ����ķֱ���Ϊ�������ؼ�1  ������ȱ�ڼ���С���ȵ�ֵ
	LineFinder() :deltaRho(1), deltaTheta(CV_PI / 180), minVote(10), minLength(0.), maxGap(0.) {};

	//�����ۼ����ķֱ���
	void setAccResolution(double dRho, double dTheta) {

		deltaRho = dRho;
		deltaTheta = dTheta;
	}

	//������СͶƱ��
	void setMinVote(int minv) {
		minVote = minv;
	}

	//����ȱ�ڼ���С����
	void setLineLengthAndGap(double length, double gap) {
		minLength = length;
		maxGap = gap;
	}

	//ʹ�ø��ʻ���任
	vector<Vec4i>findLines(Mat &binary) {

		lines.clear();
		HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
		return lines;
	}

	//���Ƽ�⵽��ֱ��
	void drawDetectedLines(Mat &image, Scalar color = Scalar(255, 255, 255)) {
		//����
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

int main() {
	Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}

	// detecting keypoints
	Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(400);
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	// computing descriptors
	Ptr<SurfDescriptorExtractor> extractor = SurfDescriptorExtractor::create();
	Mat descriptors1, descriptors2;
	extractor->compute(img1, keypoints1, descriptors1);
	extractor->compute(img2, keypoints2, descriptors2);

	waitKey(0);
	return 0;
}