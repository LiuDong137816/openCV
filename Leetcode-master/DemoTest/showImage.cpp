#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;
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

//-----------------------------------【main( )函数】--------------------------------------------
//	描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main()
{
	//改变控制台前景色和背景色
	system("color5F");

	//读入用户提供的图像
	g_srcImage = imread("girl.png");
	if (!g_srcImage.data) { printf("Oh，no，读取g_srcImage图片错误~！\n"); return false; }
	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	//设定对比度和亮度的初值
	g_nContrastValue = 80;
	g_nBrightValue = 80;

	//创建窗口
	namedWindow("【效果图窗口】", 1);

	//创建轨迹条
	createTrackbar("对比度：", "【效果图窗口】", &g_nContrastValue, 300, ContrastAndBright);
	createTrackbar("亮   度：", "【效果图窗口】", &g_nBrightValue, 200, ContrastAndBright);

	//调用回调函数
	ContrastAndBright(g_nContrastValue, 0);
	ContrastAndBright(g_nBrightValue, 0);

	//输出一些帮助信息
	cout << endl << "\t嗯。好了，请调整滚动条观察图像效果~\n\n"
		<< "\t按下“q”键时，程序退出~!\n"
		<< "\n\n\t\t\t\tby浅墨";

	//按下“q”键时，程序退出
	while (char(waitKey(1)) != 'q') {}
	return 0;
}

 
