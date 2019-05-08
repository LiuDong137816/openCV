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

int main_image() {
	Mat in_image, out_image;
	in_image = imread("1.jpg", IMREAD_UNCHANGED);
	if (in_image.empty()) {
		cout << "Error! Input image cannot be read...\n";
		return -1;
	}
	namedWindow("1", WINDOW_AUTOSIZE);
	namedWindow("2", WINDOW_AUTOSIZE);
	imshow("1", in_image);
	cvtColor(in_image, out_image, COLOR_BGR2GRAY);
	imshow("2", out_image);
	cout << "Press any key to exit...\n";
	waitKey();
	imwrite("2.jpg", in_image);
	return 0;
}

int main_showImage() {
	Mat in_frame, out_frame;
	const char win1[] = "1", win2[] = "2";
	double fps = 30;
	char fileout[] = "record.avi";

	VideoCapture inVid(0);
	if (!inVid.isOpened()) {
		cout << "Error! Camera not ready...\n";
		return -1;
	}
	int width = (int)inVid.get(CAP_PROP_FRAME_WIDTH);
	int height = (int)inVid.get(CAP_PROP_FRAME_HEIGHT);
	VideoWriter recVid(fileout, VideoWriter::fourcc('M', 'S', 'V', 'C'), fps, Size(width, height));
	if (!recVid.isOpened()) {
		cout << "Error! Video file not opened...\n";
		return -1;
	}
	namedWindow(win1);
	namedWindow(win2);
	while (true)
	{
		inVid >> in_frame;
		cvtColor(in_frame, out_frame, COLOR_BGR2GRAY);
		recVid << out_frame;
		imshow(win1, in_frame);
		imshow(win2, out_frame);
		if (waitKey(1000 / fps) >= 0)
			break;
	}
	inVid.release();
	return 0;
}

void createAlphaMat(Mat &mat)
{
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			Vec4b&rgba = mat.at<Vec4b>(i, j);
			rgba[0] = UCHAR_MAX;
			rgba[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) *UCHAR_MAX);
			rgba[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) *UCHAR_MAX);
			rgba[3] = saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));
		}
	}
}

int main_alpha()
{
	//创建带alpha通道的Mat
	Mat mat(480, 640, CV_8UC4);
	createAlphaMat(mat);

	vector<int>compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	try {
		imwrite("1.jpg", mat, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "图像转换成PNG格式发生错误：%s\n", ex.what());
		return 1;
	}

	fprintf(stdout, "PNG图片文件的alpha数据保存完毕~\n");
	system("pause");
	return 0;
}

int main_ROI()
{
	//-----------------------------------【一、图像的载入和显示】--------------------------------------
	//     描述：以下三行代码用于完成图像的载入和显示
	//--------------------------------------------------------------------------------------------------

	Mat girl = imread("girl.png"); //载入图像到Mat
	namedWindow("【1】动漫图"); //创建一个名为 "【1】动漫图"的窗口 
	imshow("【1】动漫图", girl);//显示名为 "【1】动漫图"的窗口 

	//-----------------------------------【二、初级图像混合】--------------------------------------
	//     描述：二、初级图像混合
	//-----------------------------------------------------------------------------------------------
	//载入图片
	Mat image = imread("dota.png");
	Mat logo = imread("logo.png");

	//载入后先显示
	namedWindow("【2】原画图");
	imshow("【2】原画图", image);

	namedWindow("【3】logo图");
	imshow("【3】logo图", logo);

	//定义一个Mat类型，用于存放，图像的ROI
	Mat imageROI;
	//方法一
	//imageROI = image(Rect(800, 350, logo.cols, logo.rows));
	//方法二
	imageROI=image(Range(350,350+logo.rows),Range(800,800+logo.cols));

	//将logo加到原图上
	addWeighted(imageROI, 0.5, logo, 0.3, 0., imageROI);

	//显示结果
	namedWindow("【4】原画+logo图");
	imshow("【4】原画+logo图", image);

	//-----------------------------------【三、图像的输出】--------------------------------------
	//     描述：将一个Mat图像输出到图像文件
	//-----------------------------------------------------------------------------------------------
	//输出一张jpg图片到工程目录下
	imwrite("dota1.png", image);

	waitKey();

	return 0;
}

bool ROI_AddImage()
{

	//【1】读入图像
	Mat srcImage1 = imread("dota.png");
	Mat logoImage = imread("logo.png");
	if (!srcImage1.data) { printf("你妹，读取srcImage1错误~！ \n"); return false; }
	if (!logoImage.data) { printf("你妹，读取logoImage错误~！ \n"); return false; }

	//【2】定义一个Mat类型并给其设定ROI区域
	Mat imageROI = srcImage1(Rect(200, 250, logoImage.cols, logoImage.rows));

	//【3】加载掩模（必须是灰度图）
	Mat mask = imread("logo.png", 0);

	//【4】将掩膜拷贝到ROI
	logoImage.copyTo(imageROI, mask);
	//【5】显示结果
	namedWindow("<1>利用ROI实现图像叠加示例窗口");
	imshow("<1>利用ROI实现图像叠加示例窗口", srcImage1);
	
	return true;
}

bool LinearBlending()
{
	//【0】定义一些局部变量
	double alphaValue = 0.5;
	double betaValue;
	Mat srcImage2, srcImage3, dstImage;

	//【1】读取图像 ( 两幅图片需为同样的类型和尺寸 )
	srcImage2 = imread("dota.png");
	srcImage3 = imread("dota1.png");

	if (!srcImage2.data) { printf("你妹，读取srcImage2错误~！ \n"); return false; }
	if (!srcImage3.data) { printf("你妹，读取srcImage3错误~！ \n"); return false; }

	//【2】做图像混合加权操作
	betaValue = (1.0 - alphaValue);
	addWeighted(srcImage2, alphaValue, srcImage3, betaValue, 0.0, dstImage);

	//【3】创建并显示原图窗口
	namedWindow("<2>线性混合示例窗口【原图】 by浅墨", 1);
	imshow("<2>线性混合示例窗口【原图】 by浅墨", srcImage2);

	namedWindow("<3>线性混合示例窗口【效果图】 by浅墨", 1);
	imshow("<3>线性混合示例窗口【效果图】 by浅墨", dstImage);

	return true;

}

bool ROI_LinearBlending()
{

	//【1】读取图像
	Mat srcImage4 = imread("dota.png", 1);
	Mat logoImage = imread("logo.png");

	if (!srcImage4.data) { printf("你妹，读取srcImage4错误~！ \n"); return false; }
	if (!logoImage.data) { printf("你妹，读取logoImage错误~！ \n"); return false; }

	//【2】定义一个Mat类型并给其设定ROI区域
	Mat imageROI;
	//方法一
	imageROI = srcImage4(Rect(200, 250, logoImage.cols, logoImage.rows));
	//方法二
	//imageROI=srcImage4(Range(250,250+logoImage.rows),Range(200,200+logoImage.cols));

	//【3】将logo加到原图上
	addWeighted(imageROI, 0.5, logoImage, 0.3, 0., imageROI);

	//【4】显示结果
	namedWindow("<4>区域线性图像混合示例窗口 by浅墨");
	imshow("<4>区域线性图像混合示例窗口 by浅墨", srcImage4);

	return true;
}

bool MultiChannelBlending()
{
	//【0】定义相关变量
	Mat srcImage;
	Mat logoImage;
	vector<Mat>channels;
	Mat  imageBlueChannel;

	//=================【蓝色通道部分】=================
	//     描述：多通道混合-蓝色分量部分
	//============================================

	//【1】读入图片
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh，no，读取logoImage错误~！\n"); return false; }
	if (!srcImage.data) { printf("Oh，no，读取srcImage错误~！\n"); return false; }

	//【2】把一个3通道图像转换成3个单通道图像
	split(srcImage, channels);//分离色彩通道
	//【3】将原图的蓝色通道引用返回给imageBlueChannel，注意是引用，相当于两者等价，修改其中一个另一个跟着变
	imageBlueChannel = channels.at(0);
	//【4】将原图的蓝色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，将得到的混合结果存到imageBlueChannel中
	addWeighted(imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, .5, 0, imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	//【5】将三个单通道重新合并成一个三通道
	merge(channels, srcImage);

	//【6】显示效果图
	namedWindow("<1>游戏原画+logo蓝色通道 by浅墨");
	imshow("<1>游戏原画+logo蓝色通道 by浅墨", srcImage);
	return 0;

	//=================【绿色通道部分】=================
	//     描述：多通道混合-绿色分量部分
	//============================================

	//【0】定义相关变量
	Mat  imageGreenChannel;

	//【1】重新读入图片
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh，no，读取logoImage错误~！\n"); return false; }
	if (!srcImage.data) { printf("Oh，no，读取srcImage错误~！\n"); return false; }

	//【2】将一个三通道图像转换成三个单通道图像
	split(srcImage, channels);//分离色彩通道

	//【3】将原图的绿色通道的引用返回给imageBlueChannel，注意是引用，相当于两者等价，修改其中一个另一个跟着变
	imageGreenChannel = channels.at(1);
	//【4】将原图的绿色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，将得到的混合结果存到imageGreenChannel中
	addWeighted(imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0., imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	//【5】将三个独立的单通道重新合并成一个三通道
	merge(channels, srcImage);

	//【6】显示效果图
	namedWindow("<2>游戏原画+logo绿色通道 by浅墨");
	imshow("<2>游戏原画+logo绿色通道 by浅墨", srcImage);



	//=================【红色通道部分】=================
	//     描述：多通道混合-红色分量部分
	//============================================

	//【0】定义相关变量
	Mat  imageRedChannel;

	//【1】重新读入图片
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh，no，读取logoImage错误~！\n"); return false; }
	if (!srcImage.data) { printf("Oh，no，读取srcImage错误~！\n"); return false; }

	//【2】将一个三通道图像转换成三个单通道图像
	split(srcImage, channels);//分离色彩通道

	//【3】将原图的红色通道引用返回给imageBlueChannel，注意是引用，相当于两者等价，修改其中一个另一个跟着变
	imageRedChannel = channels.at(2);
	//【4】将原图的红色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，将得到的混合结果存到imageRedChannel中
	addWeighted(imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0., imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));

	//【5】将三个独立的单通道重新合并成一个三通道
	merge(channels, srcImage);

	//【6】显示效果图
	namedWindow("<3>游戏原画+logo红色通道 by浅墨");
	imshow("<3>游戏原画+logo红色通道 by浅墨", srcImage);

	return true;
}


Mat img;
int threshval = 160;
static void on_trackbar(int, void*)
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);

	//定义点和向量
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//查找轮廓
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//初始化dst
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	//开始处理
	if (!contours.empty() && !hierarchy.empty())
	{
		//遍历所有顶层轮廓，随机生成颜色值绘制给各连接组成部分
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			//绘制填充轮廓
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}
	//显示窗口
	imshow("Connected Components", dst);
}
int g_nContrastValue; //对比度值
int g_nBrightValue;  //亮度值
Mat g_srcImage, g_dstImage;

static void ContrastAndBright(int, void *)
{
	//创建窗口
	namedWindow("【原始图窗口】", 1);

	//三个for循环，执行运算 g_dstImage(i,j) =a*g_srcImage(i,j) + b
	for (int y = 0; y < g_srcImage.rows; y++)
	{
		for (int x = 0; x < g_srcImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}

	//显示图像
	imshow("【原始图窗口】", g_srcImage);
	imshow("【效果图窗口】", g_dstImage);
}

Mat g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5;//存储图片的Mat类型
int g_nBoxFilterValue = 3;  //方框滤波参数值
int g_nMeanBlurValue = 3;  //均值滤波参数值
int g_nGaussianBlurValue = 3;  //高斯滤波参数值
int g_nMedianBlurValue = 10;  //中值滤波参数值
int g_nBilateralFilterValue = 10;  //双边滤波参数值

//-----------------------------【on_BoxFilter( )函数】------------------------------------
//     描述：方框滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BoxFilter(int, void *)
{
	//方框滤波操作
	boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	//显示窗口
	imshow("【<1>方框滤波】", g_dstImage1);
}


//-----------------------------【on_MeanBlur( )函数】------------------------------------
//     描述：均值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MeanBlur(int, void *)
{
	//均值滤波操作
	blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
	//显示窗口
	imshow("【<2>均值滤波】", g_dstImage2);
}


//-----------------------------【on_GaussianBlur( )函数】------------------------------------
//     描述：高斯滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_GaussianBlur(int, void *)
{
	//高斯滤波操作
	GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	//显示窗口
	imshow("【<3>高斯滤波】", g_dstImage3);
}

//-----------------------------【on_MedianBlur( )函数】------------------------------------
//            描述：中值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MedianBlur(int, void *)
{
	medianBlur(g_srcImage, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	imshow("【<4>中值滤波】", g_dstImage4);
}


//-----------------------------【on_BilateralFilter( )函数】------------------------------------
//            描述：双边滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BilateralFilter(int, void *)
{
	bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	imshow("【<5>双边滤波】", g_dstImage5);
}


int g_nTrackbarNumer = 0;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸

//-----------------------------【Process( )函数】------------------------------------
//            描述：进行自定义的腐蚀和膨胀操作
//-----------------------------------------------------------------------------------------
void Process()
{
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));

	//进行腐蚀或膨胀操作
	if (g_nTrackbarNumer == 0) {
		erode(g_srcImage, g_dstImage, element);
	}
	else {
		dilate(g_srcImage, g_dstImage, element);
	}

	//显示效果图
	imshow("【效果图】", g_dstImage);
}


//-----------------------------【on_TrackbarNumChange( )函数】------------------------------------
//            描述：腐蚀和膨胀之间切换开关的回调函数
//-----------------------------------------------------------------------------------------------------
void on_TrackbarNumChange(int, void *)
{
	//腐蚀和膨胀之间效果已经切换，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}


//-----------------------------【on_ElementSizeChange( )函数】-------------------------------------
//            描述：腐蚀和膨胀操作内核改变时的回调函数
//-----------------------------------------------------------------------------------------------------
void on_ElementSizeChange(int, void *)
{
	//内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
	Process();
}

int g_nElementShape = MORPH_RECT;//元素结构的形状

//变量接收的TrackBar位置参数
int g_nMaxIterationNum = 10;
int g_nOpenCloseNum = 0;
int g_nErodeDilateNum = 0;
int g_nTopBlackHatNum = 0;

//-----------------------------------【on_OpenClose( )函数】----------------------------------
//		描述：【开运算/闭运算】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*)
{
	//偏移量的定义
	int offset = g_nOpenCloseNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, CV_MOP_OPEN, element);
	else
		morphologyEx(g_srcImage, g_dstImage, CV_MOP_CLOSE, element);
	//显示图像
	imshow("【开运算/闭运算】", g_dstImage);
}


//-----------------------------------【on_ErodeDilate( )函数】----------------------------------
//		描述：【腐蚀/膨胀】窗口的回调函数
//-----------------------------------------------------------------------------------------------
static void on_ErodeDilate(int, void*)
{
	//偏移量的定义
	int offset = g_nErodeDilateNum - g_nMaxIterationNum;	//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		erode(g_srcImage, g_dstImage, element);
	else
		dilate(g_srcImage, g_dstImage, element);
	//显示图像
	imshow("【腐蚀/膨胀】", g_dstImage);
}


//-----------------------------------【on_TopBlackHat( )函数】--------------------------------
//		描述：【顶帽运算/黑帽运算】窗口的回调函数
//----------------------------------------------------------------------------------------------
static void on_TopBlackHat(int, void*)
{
	//偏移量的定义
	int offset = g_nTopBlackHatNum - g_nMaxIterationNum;//偏移量
	int Absolute_offset = offset > 0 ? offset : -offset;//偏移量绝对值
	//自定义核
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//进行操作
	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, MORPH_TOPHAT, element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_BLACKHAT, element);
	//显示图像
	imshow("【顶帽/黑帽】", g_dstImage);
}

vector<Vec4i> g_lines;//定义一个矢量结构g_lines用于存放得到的线段矢量集合
//变量接收的TrackBar位置参数
int g_nthreshold = 100;
Mat g_midImage;
//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------

static void on_HoughLines(int, void*);//回调函数

Mat g_srcGrayImage;

//Canny边缘检测相关变量
Mat g_cannyDetectedEdges;
int g_cannyLowThreshold = 1;//TrackBar位置参数  

//Sobel边缘检测相关变量
Mat g_sobelGradient_X, g_sobelGradient_Y;
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y;
int g_sobelKernelSize = 1;//TrackBar位置参数  

//Scharr滤波器相关变量
Mat g_scharrGradient_X, g_scharrGradient_Y;
Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;


//-----------------------------------【on_Canny( )函数】----------------------------------
//		描述：Canny边缘检测窗口滚动条的回调函数
//-----------------------------------------------------------------------------------------------
void on_Canny(int, void*)
{
	// 先使用 3x3内核来降噪
	blur(g_srcGrayImage, g_cannyDetectedEdges, Size(3, 3));

	// 运行我们的Canny算子
	Canny(g_cannyDetectedEdges, g_cannyDetectedEdges, g_cannyLowThreshold, g_cannyLowThreshold * 3, 3);

	//先将g_dstImage内的所有元素设置为0 
	g_dstImage = Scalar::all(0);

	//使用Canny算子输出的边缘图g_cannyDetectedEdges作为掩码，来将原图g_srcImage拷到目标图g_dstImage中
	g_srcImage.copyTo(g_dstImage, g_cannyDetectedEdges);

	//显示效果图
	imshow("【效果图】Canny边缘检测", g_dstImage);
}

//-----------------------------------【on_Sobel( )函数】----------------------------------
//		描述：Sobel边缘检测窗口滚动条的回调函数
//-----------------------------------------------------------------------------------------
void on_Sobel(int, void*)
{
	// 求 X方向梯度
	Sobel(g_srcImage, g_sobelGradient_X, CV_16S, 1, 0, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_X, g_sobelAbsGradient_X);//计算绝对值，并将结果转换成8位

	// 求Y方向梯度
	Sobel(g_srcImage, g_sobelGradient_Y, CV_16S, 0, 1, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_Y, g_sobelAbsGradient_Y);//计算绝对值，并将结果转换成8位

	// 合并梯度
	addWeighted(g_sobelAbsGradient_X, 0.5, g_sobelAbsGradient_Y, 0.5, 0, g_dstImage);

	//显示效果图
	imshow("【效果图】Sobel边缘检测", g_dstImage);

}


//-----------------------------------【Scharr( )函数】----------------------------------
//		描述：封装了Scharr边缘检测相关代码的函数
//-----------------------------------------------------------------------------------------
void Scharr()
{
	// 求 X方向梯度
	Scharr(g_srcImage, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_X, g_scharrAbsGradient_X);//计算绝对值，并将结果转换成8位

	// 求Y方向梯度
	Scharr(g_srcImage, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_Y, g_scharrAbsGradient_Y);//计算绝对值，并将结果转换成8位

	// 合并梯度
	addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, g_dstImage);

	//显示效果图
	imshow("【效果图】Scharr滤波器", g_dstImage);
}

void resize_py() {
	char ch;
	Mat g_tmpImage;
	g_srcImage = imread("girl.png");//工程目录下需要有一张名为1.jpg的测试图像，且其尺寸需被2的N次方整除，N为可以缩放的次数
	if (!g_srcImage.data) { printf("Oh，no，读取srcImage错误~！ \n"); return; }

	// 创建显示窗口
	g_tmpImage = g_srcImage;
	g_dstImage = g_tmpImage;
	namedWindow("图片", CV_WINDOW_AUTOSIZE);
	imshow("图片", g_srcImage);
	int key = 0;
	while (key = waitKey(9))
	{
		switch (key)
		{
		case '1':
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【A】被按下，开始进行基于【pyrUp】函数的图片放大：图片尺寸×2 \n");
			break;
		case '2':
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">检测到按键【W】被按下，开始进行基于【resize】函数的图片放大：图片尺寸×2 \n");
			break;
		case '3': //按键D按下，调用pyrDown函数
			pyrDown(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【D】被按下，开始进行基于【pyrDown】函数的图片缩小：图片尺寸/2\n");
			break;
					
		case  '4': //按键S按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">检测到按键【S】被按下，开始进行基于【resize】函数的图片缩小：图片尺寸/2\n");
			break;

		case '5'://按键2按下，调用resize函数
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2), (0, 0), (0, 0), 2);
			printf(">检测到按键【2】被按下，开始进行基于【resize】函数的图片缩小：图片尺寸/2\n");
			break;
		case 'q':
			return;
			break;
		}
		//经过操作后，显示变化后的图
		imshow("图片", g_dstImage);

		//将g_dstImage赋给g_tmpImage，方便下一次循环
		g_tmpImage = g_dstImage;
	}
	return;
}


static void on_HoughLines(int, void*)
{
	//定义局部变量储存全局变量
	Mat dstImage = g_dstImage.clone();
	Mat midImage = g_midImage.clone();

	//调用HoughLinesP函数
	vector<Vec4i> mylines;
	HoughLinesP(midImage, mylines, 1, CV_PI / 180, g_nthreshold + 1, 50, 10);

	//循环遍历绘制每一条线段
	for (size_t i = 0; i < mylines.size(); i++)
	{
		Vec4i l = mylines[i];
		line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 1, CV_AA);
	}
	//显示图像
	imshow("【效果图】", dstImage);
}
Mat g_grayImage, g_maskImage, g_srcImage1;
int g_nFillMode = 1;//漫水填充的模式
int g_nLowDifference = 20, g_nUpDifference = 20;//负差最大值、正差最大值
int g_nConnectivity = 4;//表示floodFill函数标识符低八位的连通值
int g_bIsColor = true;//是否为彩色图的标识符布尔值
bool g_bUseMask = false;//是否显示掩膜窗口的布尔值
int g_nNewMaskVal = 255;//新的重新绘制的像素值


//-----------------------------------【onMouse( )函数】--------------------------------------  
//      描述：鼠标消息onMouse回调函数
//---------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	// 若鼠标左键没有按下，便返回
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	//-------------------【<1>调用floodFill函数之前的参数准备部分】---------------
	Point seed = Point(x, y);
	int LowDifference = g_nFillMode == 0 ? 0 : g_nLowDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nLowDifference
	int UpDifference = g_nFillMode == 0 ? 0 : g_nUpDifference;//空范围的漫水填充，此值设为0，否则设为全局的g_nUpDifference
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) +
		(g_nFillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);//标识符的0~7位为g_nConnectivity，8~15位为g_nNewMaskVal左移8位的值，16~23位为CV_FLOODFILL_FIXED_RANGE或者0。

	//随机生成bgr值
	int b = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int g = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	int r = (unsigned)theRNG() & 255;//随机返回一个0~255之间的值
	Rect ccomp;//定义重绘区域的最小边界矩形区域

	Scalar newVal = g_bIsColor ? Scalar(b, g, r) : Scalar(r*0.299 + g * 0.587 + b * 0.114);//在重绘区域像素的新值，若是彩色图模式，取Scalar(b, g, r)；若是灰度图模式，取Scalar(r*0.299 + g*0.587 + b*0.114)

	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;//目标图的赋值
	int area;

	//--------------------【<2>正式调用floodFill函数】-----------------------------
	if (g_bUseMask)
	{
		threshold(g_maskImage, g_maskImage, 1, 128, CV_THRESH_BINARY);
		area = floodFill(dst, g_maskImage, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
		imshow("mask", g_maskImage);
	}
	else
	{
		area = floodFill(dst, seed, newVal, &ccomp, Scalar(LowDifference, LowDifference, LowDifference),
			Scalar(UpDifference, UpDifference, UpDifference), flags);
	}

	imshow("效果图", dst);
	cout << area << " 个像素被重绘\n";
}

int thresh = 30; //当前阈值
int max_thresh = 175; //最大阈值
#define WINDOW_NAME1 "【程序窗口1】"        //为窗口标题定义的宏  
#define WINDOW_NAME2 "【程序窗口2】"        //为窗口标题定义的宏  


//-----------------------------------【on_HoughLines( )函数】--------------------------------
//		描述：回调函数
//----------------------------------------------------------------------------------------------

void on_CornerHarris(int, void*)
{
	//---------------------------【1】定义一些局部变量-----------------------------
	Mat dstImage;//目标图
	Mat normImage;//归一化后的图
	Mat scaledImage;//线性变换后的八位无符号整型的图

	//---------------------------【2】初始化---------------------------------------
	//置零当前需要显示的两幅图，即清除上一次调用此函数时他们的值
	dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
	g_srcImage1 = g_srcImage.clone();

	//---------------------------【3】正式检测-------------------------------------
	//进行角点检测
	cornerHarris(g_grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);

	// 归一化与转换
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//将归一化后的图线性变换成8位无符号整型 

	//---------------------------【4】进行绘制-------------------------------------
	// 将检测到的，且符合阈值条件的角点绘制出来
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			if ((int)normImage.at<float>(j, i) > thresh + 80)
			{
				circle(g_srcImage1, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	//---------------------------【4】显示最终效果---------------------------------
	imshow(WINDOW_NAME1, g_srcImage1);
	imshow(WINDOW_NAME2, scaledImage);

}
#define WINDOW_NAME "【程序窗口】" 
Mat g_map_x, g_map_y;
//-----------------------------------【update_map( )函数】--------------------------------
//          描述：根据按键来更新map_x与map_x的值
//----------------------------------------------------------------------------------------------
int update_map(int key)
{
	//双层循环，遍历每一个像素点
	for (int j = 0; j < g_srcImage.rows; j++)
	{
		for (int i = 0; i < g_srcImage.cols; i++)
		{
			switch (key)
			{
			case '1': // 键盘【1】键按下，进行第一种重映射操作
				if (i > g_srcImage.cols*0.25 && i < g_srcImage.cols*0.75 && j > g_srcImage.rows*0.25 && j < g_srcImage.rows*0.75)
				{
					g_map_x.at<float>(j, i) = static_cast<float>(2 * (i - g_srcImage.cols*0.25) + 0.5);
					g_map_y.at<float>(j, i) = static_cast<float>(2 * (j - g_srcImage.rows*0.25) + 0.5);
				}
				else
				{
					g_map_x.at<float>(j, i) = 0;
					g_map_y.at<float>(j, i) = 0;
				}
				break;
			case '2':// 键盘【2】键按下，进行第二种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			case '3':// 键盘【3】键按下，进行第三种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(j);
				break;
			case '4':// 键盘【4】键按下，进行第四种重映射操作
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			}
		}
	}
	return 1;
}

//-----------------------------------【ShowHelpText( )函数】----------------------------------  
//      描述：输出一些帮助信息  
//----------------------------------------------------------------------------------------------  
static void ShowHelpText()
{
	//输出一些帮助信息  
	printf("\n\n\n\t欢迎来到重映射示例程序~\n\n");
	printf("\t当前使用的OpenCV版本为 OpenCV ");
	printf("\n\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】- 退出程序\n"
		"\t\t键盘按键【1】-  第一种映射方式\n"
		"\t\t键盘按键【2】- 第二种映射方式\n"
		"\t\t键盘按键【3】- 第三种映射方式\n"
		"\t\t键盘按键【4】- 第四种映射方式\n"
		"\n\n\t\t\t\t\t\t\t\t by浅墨\n\n\n"
	);
}

int main_10()
{
	//改变console字体颜色
	system("color 3F");

	ShowHelpText();

	//载入原始图和Mat变量定义   
	Mat g_srcImage = imread("girl.png");  //工程目录下应该有一张名为1.jpg的素材图

	//显示原始图  
	imshow("【原始图】", g_srcImage);

	//创建滚动条
	namedWindow("【效果图】", 1);
	createTrackbar("值", "【效果图】", &g_nthreshold, 200, on_HoughLines);

	//进行边缘检测和转化为灰度图
	Canny(g_srcImage, g_midImage, 50, 200, 3);//进行一次canny边缘检测
	cvtColor(g_midImage, g_dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图

	//调用一次回调函数，调用一次HoughLinesP函数
	on_HoughLines(g_nthreshold, 0);
	HoughLinesP(g_midImage, g_lines, 1, CV_PI / 180, 80, 50, 10);

	//显示效果图  
	imshow("【效果图】", g_dstImage);


	waitKey(0);

	return 0;
}
#define WINDOW_NAME3 "【经过Warp和Rotate后的图像】" 
//#include<opencv2/legacy/legacy.hpp>
//-----------------------------------【main( )函数】--------------------------------------------
//	描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main_11()
{
	//【0】变量的定义
	Mat src, src_gray, dst, abs_dst;

	//【1】载入原始图  
	src = imread("girl.png");  //工程目录下应该有一张名为1.jpg的素材图

	//【2】显示原始图 
	imshow("【原始图】图像Laplace变换", src);

	//【3】使用高斯滤波消除噪声
	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//【4】转换为灰度图
	//cvtColor(src, src_gray, CV_RGB2GRAY);

	//【5】使用Laplace函数
	Laplacian(src, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	//【6】计算绝对值，并将结果转换成8位
	convertScaleAbs(dst, abs_dst);

	//【7】显示效果图
	imshow("【效果图】图像Laplace变换", abs_dst);

	waitKey(0);

	//return 0;
	//【0】改变console字体颜色
	system("color 1A");

	//【0】显示欢迎和帮助文字
	ShowHelpText();

	//【1】载入素材图
	Mat srcImage1 = imread("girl.png", 1);
	Mat srcImage2 = imread("book.png", 1);
	if (!srcImage1.data || !srcImage2.data)
	{
		printf("读取图片错误，请确定目录下是否有imread函数指定的图片存在~！ \n"); return false;
	}

	//【2】使用SURF算子检测关键点
	int minHessian = 700;//SURF算法中的hessian阈值
	Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);//定义一个SurfFeatureDetector（SURF） 特征检测类对象  
	std::vector<KeyPoint> keyPoint1, keyPoints2;//vector模板类，存放任意类型的动态数组

	//【3】调用detect函数检测出SURF特征关键点，保存在vector容器中
	detector->detect(srcImage1, keyPoint1);
	detector->detect(srcImage2, keyPoints2);

	//【4】计算描述符（特征向量）
	Mat descriptors1, descriptors2;
	cv::Ptr<SURF> extractor = SURF::create();

	//SurfDescriptorExtractor extractor;
	
	extractor->compute(srcImage1, keyPoint1, descriptors1);
	extractor->compute(srcImage2, keyPoints2, descriptors2);

	//【5】使用BruteForce进行匹配
	// 实例化一个匹配器
	BFMatcher matcher(NORM_L2);
	//BruteForceMatcher<L2<float>> matcher;
	std::vector< DMatch > matches;
	//匹配两幅图中的描述子（descriptors）
	matcher.match(descriptors1, descriptors2, matches);

	//【6】绘制从两个图像中匹配出的关键点
	Mat imgMatches;
	drawMatches(srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches);//进行绘制

	//【7】显示效果图
	imshow("匹配图", imgMatches);

	waitKey(0);
	return 0;
}

void salt(Mat image, int n) {
	int i; 
	int j;
	for (int k = 0; k < n; k++) {
		i = rand() % image.cols;
		j = rand() % image.rows;

		if (image.type() == CV_8UC1) {
			image.at<uchar>(j, i) = 255;
		}
		else if (image.type() == CV_8UC3) {
			image.at<Vec3b>(j, i)[0] = 255;
			image.at<Vec3b>(j, i)[1] = 255;
			image.at<Vec3b>(j, i)[2] = 255;
		}
	}
}

void colorReduce(const Mat image, Mat result, int div = 64) {
	int nl = image.rows;
	int nc = image.cols * image.channels();
	for (int j = 0; j < nl; j++) {
		const uchar* indata = image.ptr<uchar>(j);
		uchar* outdata = result.ptr<uchar>(j);
		for (int i = 0; i < nc; i++) {
			outdata[i] = indata[i] / div * div + div / 2;
		}
	}
}

void sharpen(const Mat &image, Mat &result) {
	result.create(image.size(), image.type());
	int nchanels = image.channels();
	for (int j = 1; j < image.rows - 1; j++) {
		const uchar* pre = image.ptr<const uchar>(j - 1);
		const uchar* curr = image.ptr<const uchar>(j);
		const uchar* next = image.ptr<const uchar>(j + 1);
		uchar* output = result.ptr<uchar>(j);
		for(int i = nchanels; i < (image.cols - 1)* nchanels; i++) {
			*output++ = saturate_cast<uchar>(5 * curr[i] - curr[i - nchanels] - curr[i + nchanels] - pre[i] - next[i]);
		}
	}
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void sharpen2D(const Mat &image, Mat &result) {
	Mat kernel(3, 3, CV_32F, Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	filter2D(image, result, image.depth(), kernel);
}

void wave(const Mat& image, Mat result) {
	Mat srcX(image.rows, image.cols, CV_32F);
	Mat srcY(image.rows, image.cols, CV_32F);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			srcX.at<float>(i, j) = image.cols - j - 1;
			srcY.at<float>(i, j) = i;
		}
	}
	remap(image, result, srcX, srcY, INTER_LINEAR);
}

class ColorDector
{
public:
	ColorDector() :maxDist(50), target(0, 0, 0) {};
	void setTargetColor(uchar blue, uchar green, uchar red) {
		target = Vec3b(blue, green, red);
	}
	int getColorDistance(const Vec3b& color1, const Vec3b& color2)const {
		return abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2]);
	}
	int getDistanceToTargetColor(const Vec3b& color) {
		return getColorDistance(color, target);
	}
	Mat process(const Mat &image);
private:
	int maxDist;
	Vec3b target;
	Mat result;
};

Mat ColorDector::process(const Mat &image) {
	Mat output;
	absdiff(image, Scalar(target), output);
	vector<Mat> images;
	split(output, images);
	output = images[0] + images[1] + images[2];
	threshold(output, output, maxDist, 255, THRESH_BINARY_INV);
	return output;
}

void detectHScolor(const Mat& image, double minHue, double maxHue, double minSat, double maxSat, Mat& mask) {
	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);
	Mat mask1;
	threshold(channels[0], mask1, maxHue, 255, THRESH_BINARY_INV);
	Mat mask2;
	threshold(channels[0], mask2, minHue, 255, THRESH_BINARY);

	Mat hueMask;
	if (minHue < maxHue)
		hueMask = mask1 & mask2;
	else
		hueMask = mask1 | mask2;
	threshold(channels[1], mask1, maxSat, 255, THRESH_BINARY_INV);
	threshold(channels[1], mask2, minSat, 255, THRESH_BINARY);
	Mat satMask = mask1 & mask2;
	mask = hueMask & satMask;
}

class Histogram1D {
private:
	int histSize[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];
public:
	Histogram1D() {
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 256.0;
		ranges[0] = hranges;
		channels[0] = 0;
	}
	Mat getHistogram(const Mat &image);
	Mat getHistogramImage(const Mat &image, int zoom);
	static Mat getImageOfHistogram(const Mat &hist, int zoom);
	Mat strech(const Mat &image, int minValue);
	Mat applyLookUp(const Mat &image, const Mat& lookup);
};

Mat Histogram1D::getHistogram(const Mat &image) {
	Mat hist;
	calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
	return hist;
}

Mat Histogram1D::getImageOfHistogram(const Mat &hist, int zoom) {
	double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histSize = hist.rows;
	Mat histImg(histSize*zoom, histSize*zoom, CV_8U, Scalar(255));
	int hpt = static_cast<int>(0.9*histSize);
	for (int h = 0; h < histSize; h++) {
		float binVal = hist.at<float>(h);
		if (binVal > 0) {
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			line(histImg, Point(h*zoom, histSize*zoom), Point(h*zoom, (histSize - intensity)*zoom), Scalar(0), zoom);
		}
	}
	return histImg;
}

Mat Histogram1D::getHistogramImage(const Mat &image, int zoom = 1) {
	Mat hist = getHistogram(image);
	return getImageOfHistogram(hist, zoom);
}

Mat Histogram1D::applyLookUp(const Mat &image, const Mat& lookup) {
	Mat result;
	LUT(image, lookup, result);
	return result;
}

Mat Histogram1D::strech(const Mat &image, int minValue = 0) {
	Mat hist = getHistogram(image);
	int imin = 0;
	for (; imin < histSize[0]; imin++) {
		if (hist.at<float>(imin) > minValue)
			break;
	}
	int imax = histSize[0] - 1;
	for (; imax >= 0; imax--) {
		if (hist.at<float>(imax) > minValue)
			break;
	}
	int dim(256);
	Mat lookup(1, &dim, CV_8U);
	for (int i = 0; i < 256; i++) {
		if (i < imin)
			lookup.at<uchar>(i) = 0;
		else if (i > imax)
			lookup.at<uchar>(i) = 255;
		else
			lookup.at<uchar>(i) = cvRound(255.0*(i - imin) / (imax - imin));
	}
	Mat result;
	result = applyLookUp(image, lookup);
	return result;
}

class ColorHistogram
{
public:
	ColorHistogram();
private:
	int histSize[3];
	float hrangs[2];
};

class ContentFinder {
private:
	float hrange[2];
	const float* ranges[3];
	int channels[3];
	float threshold;
	Mat histogram;
public:
	ContentFinder() :threshold(0.1f) {
		ranges[0] = hrange;
		ranges[1] = hrange;
		ranges[2] = hrange;
	}
	void setThreshold(float t) {
		threshold = t;
	}
	float getThreshold() {
		return threshold;
	}
	void setHisogram(const Mat &h) {
		histogram = h;
		normalize(histogram, histogram, 1.0);
	}
	Mat find(const Mat& image);
	Mat find(const Mat& image, float minValue, float maxValue, int *channels);
};
Mat ContentFinder::find(const Mat& image, float minValue, float maxValue, int *channel) {
	Mat result;
	hrange[0] = minValue;
	hrange[1] = maxValue;
	for (int i = 0; i < histogram.dims; i++)
		this->channels[i] = channels[i];
	calcBackProject(&image, 1, channels, histogram, result, ranges, 255.0);
	if (threshold > 0.0)
		cv::threshold(result, result, 255.0*threshold, 255.0, THRESH_BINARY);
	return result;
}

Mat ContentFinder::find(const Mat& image) {
	Mat result;
	hrange[0] = 0.0;
	hrange[1] = 256.0;
	channels[0] = 0;
	channels[1] = 1;
	channels[2] = 2;
	return find(image, hrange[0], hrange[1], channels);
}

int main() {
	
	Mat image = imread("girl.png");
	ContentFinder finder;
	finder.setHisogram(image);
	finder.setThreshold(0.05f);
	Mat result = finder.find(image);
	
	Mat mask;
	detectHScolor(image, 160, 10, 25, 166, mask);
	Mat detected(image.size(), CV_8UC3, Scalar(0, 0, 0));
	image.copyTo(detected, mask);
	imshow("show1", mask);
	imshow("show", detected);
	waitKey(0);
	return 0;
}

 
