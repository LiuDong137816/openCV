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
	//������alphaͨ����Mat
	Mat mat(480, 640, CV_8UC4);
	createAlphaMat(mat);

	vector<int>compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	try {
		imwrite("1.jpg", mat, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "ͼ��ת����PNG��ʽ��������%s\n", ex.what());
		return 1;
	}

	fprintf(stdout, "PNGͼƬ�ļ���alpha���ݱ������~\n");
	system("pause");
	return 0;
}

int main_ROI()
{
	//-----------------------------------��һ��ͼ����������ʾ��--------------------------------------
	//     �������������д����������ͼ����������ʾ
	//--------------------------------------------------------------------------------------------------

	Mat girl = imread("girl.png"); //����ͼ��Mat
	namedWindow("��1������ͼ"); //����һ����Ϊ "��1������ͼ"�Ĵ��� 
	imshow("��1������ͼ", girl);//��ʾ��Ϊ "��1������ͼ"�Ĵ��� 

	//-----------------------------------����������ͼ���ϡ�--------------------------------------
	//     ��������������ͼ����
	//-----------------------------------------------------------------------------------------------
	//����ͼƬ
	Mat image = imread("dota.png");
	Mat logo = imread("logo.png");

	//���������ʾ
	namedWindow("��2��ԭ��ͼ");
	imshow("��2��ԭ��ͼ", image);

	namedWindow("��3��logoͼ");
	imshow("��3��logoͼ", logo);

	//����һ��Mat���ͣ����ڴ�ţ�ͼ���ROI
	Mat imageROI;
	//����һ
	//imageROI = image(Rect(800, 350, logo.cols, logo.rows));
	//������
	imageROI=image(Range(350,350+logo.rows),Range(800,800+logo.cols));

	//��logo�ӵ�ԭͼ��
	addWeighted(imageROI, 0.5, logo, 0.3, 0., imageROI);

	//��ʾ���
	namedWindow("��4��ԭ��+logoͼ");
	imshow("��4��ԭ��+logoͼ", image);

	//-----------------------------------������ͼ��������--------------------------------------
	//     ��������һ��Matͼ�������ͼ���ļ�
	//-----------------------------------------------------------------------------------------------
	//���һ��jpgͼƬ������Ŀ¼��
	imwrite("dota1.png", image);

	waitKey();

	return 0;
}

bool ROI_AddImage()
{

	//��1������ͼ��
	Mat srcImage1 = imread("dota.png");
	Mat logoImage = imread("logo.png");
	if (!srcImage1.data) { printf("���ã���ȡsrcImage1����~�� \n"); return false; }
	if (!logoImage.data) { printf("���ã���ȡlogoImage����~�� \n"); return false; }

	//��2������һ��Mat���Ͳ������趨ROI����
	Mat imageROI = srcImage1(Rect(200, 250, logoImage.cols, logoImage.rows));

	//��3��������ģ�������ǻҶ�ͼ��
	Mat mask = imread("logo.png", 0);

	//��4������Ĥ������ROI
	logoImage.copyTo(imageROI, mask);
	//��5����ʾ���
	namedWindow("<1>����ROIʵ��ͼ�����ʾ������");
	imshow("<1>����ROIʵ��ͼ�����ʾ������", srcImage1);
	
	return true;
}

bool LinearBlending()
{
	//��0������һЩ�ֲ�����
	double alphaValue = 0.5;
	double betaValue;
	Mat srcImage2, srcImage3, dstImage;

	//��1����ȡͼ�� ( ����ͼƬ��Ϊͬ�������ͺͳߴ� )
	srcImage2 = imread("dota.png");
	srcImage3 = imread("dota1.png");

	if (!srcImage2.data) { printf("���ã���ȡsrcImage2����~�� \n"); return false; }
	if (!srcImage3.data) { printf("���ã���ȡsrcImage3����~�� \n"); return false; }

	//��2����ͼ���ϼ�Ȩ����
	betaValue = (1.0 - alphaValue);
	addWeighted(srcImage2, alphaValue, srcImage3, betaValue, 0.0, dstImage);

	//��3����������ʾԭͼ����
	namedWindow("<2>���Ի��ʾ�����ڡ�ԭͼ�� byǳī", 1);
	imshow("<2>���Ի��ʾ�����ڡ�ԭͼ�� byǳī", srcImage2);

	namedWindow("<3>���Ի��ʾ�����ڡ�Ч��ͼ�� byǳī", 1);
	imshow("<3>���Ի��ʾ�����ڡ�Ч��ͼ�� byǳī", dstImage);

	return true;

}

bool ROI_LinearBlending()
{

	//��1����ȡͼ��
	Mat srcImage4 = imread("dota.png", 1);
	Mat logoImage = imread("logo.png");

	if (!srcImage4.data) { printf("���ã���ȡsrcImage4����~�� \n"); return false; }
	if (!logoImage.data) { printf("���ã���ȡlogoImage����~�� \n"); return false; }

	//��2������һ��Mat���Ͳ������趨ROI����
	Mat imageROI;
	//����һ
	imageROI = srcImage4(Rect(200, 250, logoImage.cols, logoImage.rows));
	//������
	//imageROI=srcImage4(Range(250,250+logoImage.rows),Range(200,200+logoImage.cols));

	//��3����logo�ӵ�ԭͼ��
	addWeighted(imageROI, 0.5, logoImage, 0.3, 0., imageROI);

	//��4����ʾ���
	namedWindow("<4>��������ͼ����ʾ������ byǳī");
	imshow("<4>��������ͼ����ʾ������ byǳī", srcImage4);

	return true;
}

bool MultiChannelBlending()
{
	//��0��������ر���
	Mat srcImage;
	Mat logoImage;
	vector<Mat>channels;
	Mat  imageBlueChannel;

	//=================����ɫͨ�����֡�=================
	//     ��������ͨ�����-��ɫ��������
	//============================================

	//��1������ͼƬ
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh��no����ȡlogoImage����~��\n"); return false; }
	if (!srcImage.data) { printf("Oh��no����ȡsrcImage����~��\n"); return false; }

	//��2����һ��3ͨ��ͼ��ת����3����ͨ��ͼ��
	split(srcImage, channels);//����ɫ��ͨ��
	//��3����ԭͼ����ɫͨ�����÷��ظ�imageBlueChannel��ע�������ã��൱�����ߵȼۣ��޸�����һ����һ�����ű�
	imageBlueChannel = channels.at(0);
	//��4����ԭͼ����ɫͨ���ģ�500,250�����괦���·���һ�������logoͼ���м�Ȩ���������õ��Ļ�Ͻ���浽imageBlueChannel��
	addWeighted(imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, .5, 0, imageBlueChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	//��5����������ͨ�����ºϲ���һ����ͨ��
	merge(channels, srcImage);

	//��6����ʾЧ��ͼ
	namedWindow("<1>��Ϸԭ��+logo��ɫͨ�� byǳī");
	imshow("<1>��Ϸԭ��+logo��ɫͨ�� byǳī", srcImage);
	return 0;

	//=================����ɫͨ�����֡�=================
	//     ��������ͨ�����-��ɫ��������
	//============================================

	//��0��������ر���
	Mat  imageGreenChannel;

	//��1�����¶���ͼƬ
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh��no����ȡlogoImage����~��\n"); return false; }
	if (!srcImage.data) { printf("Oh��no����ȡsrcImage����~��\n"); return false; }

	//��2����һ����ͨ��ͼ��ת����������ͨ��ͼ��
	split(srcImage, channels);//����ɫ��ͨ��

	//��3����ԭͼ����ɫͨ�������÷��ظ�imageBlueChannel��ע�������ã��൱�����ߵȼۣ��޸�����һ����һ�����ű�
	imageGreenChannel = channels.at(1);
	//��4����ԭͼ����ɫͨ���ģ�500,250�����괦���·���һ�������logoͼ���м�Ȩ���������õ��Ļ�Ͻ���浽imageGreenChannel��
	addWeighted(imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0., imageGreenChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));
	//��5�������������ĵ�ͨ�����ºϲ���һ����ͨ��
	merge(channels, srcImage);

	//��6����ʾЧ��ͼ
	namedWindow("<2>��Ϸԭ��+logo��ɫͨ�� byǳī");
	imshow("<2>��Ϸԭ��+logo��ɫͨ�� byǳī", srcImage);



	//=================����ɫͨ�����֡�=================
	//     ��������ͨ�����-��ɫ��������
	//============================================

	//��0��������ر���
	Mat  imageRedChannel;

	//��1�����¶���ͼƬ
	logoImage = imread("logo.png", 0);
	srcImage = imread("dota.png");

	if (!logoImage.data) { printf("Oh��no����ȡlogoImage����~��\n"); return false; }
	if (!srcImage.data) { printf("Oh��no����ȡsrcImage����~��\n"); return false; }

	//��2����һ����ͨ��ͼ��ת����������ͨ��ͼ��
	split(srcImage, channels);//����ɫ��ͨ��

	//��3����ԭͼ�ĺ�ɫͨ�����÷��ظ�imageBlueChannel��ע�������ã��൱�����ߵȼۣ��޸�����һ����һ�����ű�
	imageRedChannel = channels.at(2);
	//��4����ԭͼ�ĺ�ɫͨ���ģ�500,250�����괦���·���һ�������logoͼ���м�Ȩ���������õ��Ļ�Ͻ���浽imageRedChannel��
	addWeighted(imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)), 1.0,
		logoImage, 0.5, 0., imageRedChannel(Rect(500, 250, logoImage.cols, logoImage.rows)));

	//��5�������������ĵ�ͨ�����ºϲ���һ����ͨ��
	merge(channels, srcImage);

	//��6����ʾЧ��ͼ
	namedWindow("<3>��Ϸԭ��+logo��ɫͨ�� byǳī");
	imshow("<3>��Ϸԭ��+logo��ɫͨ�� byǳī", srcImage);

	return true;
}


Mat img;
int threshval = 160;
static void on_trackbar(int, void*)
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);

	//����������
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//��������
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//��ʼ��dst
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	//��ʼ����
	if (!contours.empty() && !hierarchy.empty())
	{
		//�������ж������������������ɫֵ���Ƹ���������ɲ���
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			//�����������
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}
	//��ʾ����
	imshow("Connected Components", dst);
}
int g_nContrastValue; //�Աȶ�ֵ
int g_nBrightValue;  //����ֵ
Mat g_srcImage, g_dstImage;

static void ContrastAndBright(int, void *)
{
	//��������
	namedWindow("��ԭʼͼ���ڡ�", 1);

	//����forѭ����ִ������ g_dstImage(i,j) =a*g_srcImage(i,j) + b
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

	//��ʾͼ��
	imshow("��ԭʼͼ���ڡ�", g_srcImage);
	imshow("��Ч��ͼ���ڡ�", g_dstImage);
}

Mat g_dstImage1, g_dstImage2, g_dstImage3, g_dstImage4, g_dstImage5;//�洢ͼƬ��Mat����
int g_nBoxFilterValue = 3;  //�����˲�����ֵ
int g_nMeanBlurValue = 3;  //��ֵ�˲�����ֵ
int g_nGaussianBlurValue = 3;  //��˹�˲�����ֵ
int g_nMedianBlurValue = 10;  //��ֵ�˲�����ֵ
int g_nBilateralFilterValue = 10;  //˫���˲�����ֵ

//-----------------------------��on_BoxFilter( )������------------------------------------
//     �����������˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_BoxFilter(int, void *)
{
	//�����˲�����
	boxFilter(g_srcImage, g_dstImage1, -1, Size(g_nBoxFilterValue + 1, g_nBoxFilterValue + 1));
	//��ʾ����
	imshow("��<1>�����˲���", g_dstImage1);
}


//-----------------------------��on_MeanBlur( )������------------------------------------
//     ��������ֵ�˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_MeanBlur(int, void *)
{
	//��ֵ�˲�����
	blur(g_srcImage, g_dstImage2, Size(g_nMeanBlurValue + 1, g_nMeanBlurValue + 1), Point(-1, -1));
	//��ʾ����
	imshow("��<2>��ֵ�˲���", g_dstImage2);
}


//-----------------------------��on_GaussianBlur( )������------------------------------------
//     ��������˹�˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_GaussianBlur(int, void *)
{
	//��˹�˲�����
	GaussianBlur(g_srcImage, g_dstImage3, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);
	//��ʾ����
	imshow("��<3>��˹�˲���", g_dstImage3);
}

//-----------------------------��on_MedianBlur( )������------------------------------------
//            ��������ֵ�˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_MedianBlur(int, void *)
{
	medianBlur(g_srcImage, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	imshow("��<4>��ֵ�˲���", g_dstImage4);
}


//-----------------------------��on_BilateralFilter( )������------------------------------------
//            ������˫���˲������Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_BilateralFilter(int, void *)
{
	bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	imshow("��<5>˫���˲���", g_dstImage5);
}


int g_nTrackbarNumer = 0;//0��ʾ��ʴerode, 1��ʾ����dilate
int g_nStructElementSize = 3; //�ṹԪ��(�ں˾���)�ĳߴ�

//-----------------------------��Process( )������------------------------------------
//            �����������Զ���ĸ�ʴ�����Ͳ���
//-----------------------------------------------------------------------------------------
void Process()
{
	//��ȡ�Զ����
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));

	//���и�ʴ�����Ͳ���
	if (g_nTrackbarNumer == 0) {
		erode(g_srcImage, g_dstImage, element);
	}
	else {
		dilate(g_srcImage, g_dstImage, element);
	}

	//��ʾЧ��ͼ
	imshow("��Ч��ͼ��", g_dstImage);
}


//-----------------------------��on_TrackbarNumChange( )������------------------------------------
//            ��������ʴ������֮���л����صĻص�����
//-----------------------------------------------------------------------------------------------------
void on_TrackbarNumChange(int, void *)
{
	//��ʴ������֮��Ч���Ѿ��л����ص��������������һ��Process������ʹ�ı���Ч��������Ч����ʾ����
	Process();
}


//-----------------------------��on_ElementSizeChange( )������-------------------------------------
//            ��������ʴ�����Ͳ����ں˸ı�ʱ�Ļص�����
//-----------------------------------------------------------------------------------------------------
void on_ElementSizeChange(int, void *)
{
	//�ں˳ߴ��Ѹı䣬�ص��������������һ��Process������ʹ�ı���Ч��������Ч����ʾ����
	Process();
}

int g_nElementShape = MORPH_RECT;//Ԫ�ؽṹ����״

//�������յ�TrackBarλ�ò���
int g_nMaxIterationNum = 10;
int g_nOpenCloseNum = 0;
int g_nErodeDilateNum = 0;
int g_nTopBlackHatNum = 0;

//-----------------------------------��on_OpenClose( )������----------------------------------
//		��������������/�����㡿���ڵĻص�����
//-----------------------------------------------------------------------------------------------
static void on_OpenClose(int, void*)
{
	//ƫ�����Ķ���
	int offset = g_nOpenCloseNum - g_nMaxIterationNum;//ƫ����
	int Absolute_offset = offset > 0 ? offset : -offset;//ƫ��������ֵ
	//�Զ����
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//���в���
	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, CV_MOP_OPEN, element);
	else
		morphologyEx(g_srcImage, g_dstImage, CV_MOP_CLOSE, element);
	//��ʾͼ��
	imshow("��������/�����㡿", g_dstImage);
}


//-----------------------------------��on_ErodeDilate( )������----------------------------------
//		����������ʴ/���͡����ڵĻص�����
//-----------------------------------------------------------------------------------------------
static void on_ErodeDilate(int, void*)
{
	//ƫ�����Ķ���
	int offset = g_nErodeDilateNum - g_nMaxIterationNum;	//ƫ����
	int Absolute_offset = offset > 0 ? offset : -offset;//ƫ��������ֵ
	//�Զ����
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//���в���
	if (offset < 0)
		erode(g_srcImage, g_dstImage, element);
	else
		dilate(g_srcImage, g_dstImage, element);
	//��ʾͼ��
	imshow("����ʴ/���͡�", g_dstImage);
}


//-----------------------------------��on_TopBlackHat( )������--------------------------------
//		����������ñ����/��ñ���㡿���ڵĻص�����
//----------------------------------------------------------------------------------------------
static void on_TopBlackHat(int, void*)
{
	//ƫ�����Ķ���
	int offset = g_nTopBlackHatNum - g_nMaxIterationNum;//ƫ����
	int Absolute_offset = offset > 0 ? offset : -offset;//ƫ��������ֵ
	//�Զ����
	Mat element = getStructuringElement(g_nElementShape, Size(Absolute_offset * 2 + 1, Absolute_offset * 2 + 1), Point(Absolute_offset, Absolute_offset));
	//���в���
	if (offset < 0)
		morphologyEx(g_srcImage, g_dstImage, MORPH_TOPHAT, element);
	else
		morphologyEx(g_srcImage, g_dstImage, MORPH_BLACKHAT, element);
	//��ʾͼ��
	imshow("����ñ/��ñ��", g_dstImage);
}

vector<Vec4i> g_lines;//����һ��ʸ���ṹg_lines���ڴ�ŵõ����߶�ʸ������
//�������յ�TrackBarλ�ò���
int g_nthreshold = 100;
Mat g_midImage;
//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//		������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------

static void on_HoughLines(int, void*);//�ص�����

Mat g_srcGrayImage;

//Canny��Ե�����ر���
Mat g_cannyDetectedEdges;
int g_cannyLowThreshold = 1;//TrackBarλ�ò���  

//Sobel��Ե�����ر���
Mat g_sobelGradient_X, g_sobelGradient_Y;
Mat g_sobelAbsGradient_X, g_sobelAbsGradient_Y;
int g_sobelKernelSize = 1;//TrackBarλ�ò���  

//Scharr�˲�����ر���
Mat g_scharrGradient_X, g_scharrGradient_Y;
Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;


//-----------------------------------��on_Canny( )������----------------------------------
//		������Canny��Ե��ⴰ�ڹ������Ļص�����
//-----------------------------------------------------------------------------------------------
void on_Canny(int, void*)
{
	// ��ʹ�� 3x3�ں�������
	blur(g_srcGrayImage, g_cannyDetectedEdges, Size(3, 3));

	// �������ǵ�Canny����
	Canny(g_cannyDetectedEdges, g_cannyDetectedEdges, g_cannyLowThreshold, g_cannyLowThreshold * 3, 3);

	//�Ƚ�g_dstImage�ڵ�����Ԫ������Ϊ0 
	g_dstImage = Scalar::all(0);

	//ʹ��Canny��������ı�Եͼg_cannyDetectedEdges��Ϊ���룬����ԭͼg_srcImage����Ŀ��ͼg_dstImage��
	g_srcImage.copyTo(g_dstImage, g_cannyDetectedEdges);

	//��ʾЧ��ͼ
	imshow("��Ч��ͼ��Canny��Ե���", g_dstImage);
}

//-----------------------------------��on_Sobel( )������----------------------------------
//		������Sobel��Ե��ⴰ�ڹ������Ļص�����
//-----------------------------------------------------------------------------------------
void on_Sobel(int, void*)
{
	// �� X�����ݶ�
	Sobel(g_srcImage, g_sobelGradient_X, CV_16S, 1, 0, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_X, g_sobelAbsGradient_X);//�������ֵ���������ת����8λ

	// ��Y�����ݶ�
	Sobel(g_srcImage, g_sobelGradient_Y, CV_16S, 0, 1, (2 * g_sobelKernelSize + 1), 1, 1, BORDER_DEFAULT);
	convertScaleAbs(g_sobelGradient_Y, g_sobelAbsGradient_Y);//�������ֵ���������ת����8λ

	// �ϲ��ݶ�
	addWeighted(g_sobelAbsGradient_X, 0.5, g_sobelAbsGradient_Y, 0.5, 0, g_dstImage);

	//��ʾЧ��ͼ
	imshow("��Ч��ͼ��Sobel��Ե���", g_dstImage);

}


//-----------------------------------��Scharr( )������----------------------------------
//		��������װ��Scharr��Ե�����ش���ĺ���
//-----------------------------------------------------------------------------------------
void Scharr()
{
	// �� X�����ݶ�
	Scharr(g_srcImage, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_X, g_scharrAbsGradient_X);//�������ֵ���������ת����8λ

	// ��Y�����ݶ�
	Scharr(g_srcImage, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(g_scharrGradient_Y, g_scharrAbsGradient_Y);//�������ֵ���������ת����8λ

	// �ϲ��ݶ�
	addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, g_dstImage);

	//��ʾЧ��ͼ
	imshow("��Ч��ͼ��Scharr�˲���", g_dstImage);
}

void resize_py() {
	char ch;
	Mat g_tmpImage;
	g_srcImage = imread("girl.png");//����Ŀ¼����Ҫ��һ����Ϊ1.jpg�Ĳ���ͼ������ߴ��豻2��N�η�������NΪ�������ŵĴ���
	if (!g_srcImage.data) { printf("Oh��no����ȡsrcImage����~�� \n"); return; }

	// ������ʾ����
	g_tmpImage = g_srcImage;
	g_dstImage = g_tmpImage;
	namedWindow("ͼƬ", CV_WINDOW_AUTOSIZE);
	imshow("ͼƬ", g_srcImage);
	int key = 0;
	while (key = waitKey(9))
	{
		switch (key)
		{
		case '1':
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">��⵽������A�������£���ʼ���л��ڡ�pyrUp��������ͼƬ�Ŵ�ͼƬ�ߴ��2 \n");
			break;
		case '2':
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">��⵽������W�������£���ʼ���л��ڡ�resize��������ͼƬ�Ŵ�ͼƬ�ߴ��2 \n");
			break;
		case '3': //����D���£�����pyrDown����
			pyrDown(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">��⵽������D�������£���ʼ���л��ڡ�pyrDown��������ͼƬ��С��ͼƬ�ߴ�/2\n");
			break;
					
		case  '4': //����S���£�����resize����
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2));
			printf(">��⵽������S�������£���ʼ���л��ڡ�resize��������ͼƬ��С��ͼƬ�ߴ�/2\n");
			break;

		case '5'://����2���£�����resize����
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols / 2, g_tmpImage.rows / 2), (0, 0), (0, 0), 2);
			printf(">��⵽������2�������£���ʼ���л��ڡ�resize��������ͼƬ��С��ͼƬ�ߴ�/2\n");
			break;
		case 'q':
			return;
			break;
		}
		//������������ʾ�仯���ͼ
		imshow("ͼƬ", g_dstImage);

		//��g_dstImage����g_tmpImage��������һ��ѭ��
		g_tmpImage = g_dstImage;
	}
	return;
}


static void on_HoughLines(int, void*)
{
	//����ֲ���������ȫ�ֱ���
	Mat dstImage = g_dstImage.clone();
	Mat midImage = g_midImage.clone();

	//����HoughLinesP����
	vector<Vec4i> mylines;
	HoughLinesP(midImage, mylines, 1, CV_PI / 180, g_nthreshold + 1, 50, 10);

	//ѭ����������ÿһ���߶�
	for (size_t i = 0; i < mylines.size(); i++)
	{
		Vec4i l = mylines[i];
		line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 1, CV_AA);
	}
	//��ʾͼ��
	imshow("��Ч��ͼ��", dstImage);
}
Mat g_grayImage, g_maskImage, g_srcImage1;
int g_nFillMode = 1;//��ˮ����ģʽ
int g_nLowDifference = 20, g_nUpDifference = 20;//�������ֵ���������ֵ
int g_nConnectivity = 4;//��ʾfloodFill������ʶ���Ͱ�λ����ֵͨ
int g_bIsColor = true;//�Ƿ�Ϊ��ɫͼ�ı�ʶ������ֵ
bool g_bUseMask = false;//�Ƿ���ʾ��Ĥ���ڵĲ���ֵ
int g_nNewMaskVal = 255;//�µ����»��Ƶ�����ֵ


//-----------------------------------��onMouse( )������--------------------------------------  
//      �����������ϢonMouse�ص�����
//---------------------------------------------------------------------------------------------
static void onMouse(int event, int x, int y, int, void*)
{
	// ��������û�а��£��㷵��
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	//-------------------��<1>����floodFill����֮ǰ�Ĳ���׼�����֡�---------------
	Point seed = Point(x, y);
	int LowDifference = g_nFillMode == 0 ? 0 : g_nLowDifference;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nLowDifference
	int UpDifference = g_nFillMode == 0 ? 0 : g_nUpDifference;//�շ�Χ����ˮ��䣬��ֵ��Ϊ0��������Ϊȫ�ֵ�g_nUpDifference
	int flags = g_nConnectivity + (g_nNewMaskVal << 8) +
		(g_nFillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);//��ʶ����0~7λΪg_nConnectivity��8~15λΪg_nNewMaskVal����8λ��ֵ��16~23λΪCV_FLOODFILL_FIXED_RANGE����0��

	//�������bgrֵ
	int b = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	int g = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	int r = (unsigned)theRNG() & 255;//�������һ��0~255֮���ֵ
	Rect ccomp;//�����ػ��������С�߽��������

	Scalar newVal = g_bIsColor ? Scalar(b, g, r) : Scalar(r*0.299 + g * 0.587 + b * 0.114);//���ػ��������ص���ֵ�����ǲ�ɫͼģʽ��ȡScalar(b, g, r)�����ǻҶ�ͼģʽ��ȡScalar(r*0.299 + g*0.587 + b*0.114)

	Mat dst = g_bIsColor ? g_dstImage : g_grayImage;//Ŀ��ͼ�ĸ�ֵ
	int area;

	//--------------------��<2>��ʽ����floodFill������-----------------------------
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

	imshow("Ч��ͼ", dst);
	cout << area << " �����ر��ػ�\n";
}

int thresh = 30; //��ǰ��ֵ
int max_thresh = 175; //�����ֵ
#define WINDOW_NAME1 "�����򴰿�1��"        //Ϊ���ڱ��ⶨ��ĺ�  
#define WINDOW_NAME2 "�����򴰿�2��"        //Ϊ���ڱ��ⶨ��ĺ�  


//-----------------------------------��on_HoughLines( )������--------------------------------
//		�������ص�����
//----------------------------------------------------------------------------------------------

void on_CornerHarris(int, void*)
{
	//---------------------------��1������һЩ�ֲ�����-----------------------------
	Mat dstImage;//Ŀ��ͼ
	Mat normImage;//��һ�����ͼ
	Mat scaledImage;//���Ա任��İ�λ�޷������͵�ͼ

	//---------------------------��2����ʼ��---------------------------------------
	//���㵱ǰ��Ҫ��ʾ������ͼ���������һ�ε��ô˺���ʱ���ǵ�ֵ
	dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
	g_srcImage1 = g_srcImage.clone();

	//---------------------------��3����ʽ���-------------------------------------
	//���нǵ���
	cornerHarris(g_grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);

	// ��һ����ת��
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//����һ�����ͼ���Ա任��8λ�޷������� 

	//---------------------------��4�����л���-------------------------------------
	// ����⵽�ģ��ҷ�����ֵ�����Ľǵ���Ƴ���
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
	//---------------------------��4����ʾ����Ч��---------------------------------
	imshow(WINDOW_NAME1, g_srcImage1);
	imshow(WINDOW_NAME2, scaledImage);

}
#define WINDOW_NAME "�����򴰿ڡ�" 
Mat g_map_x, g_map_y;
//-----------------------------------��update_map( )������--------------------------------
//          ���������ݰ���������map_x��map_x��ֵ
//----------------------------------------------------------------------------------------------
int update_map(int key)
{
	//˫��ѭ��������ÿһ�����ص�
	for (int j = 0; j < g_srcImage.rows; j++)
	{
		for (int i = 0; i < g_srcImage.cols; i++)
		{
			switch (key)
			{
			case '1': // ���̡�1�������£����е�һ����ӳ�����
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
			case '2':// ���̡�2�������£����еڶ�����ӳ�����
				g_map_x.at<float>(j, i) = static_cast<float>(i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			case '3':// ���̡�3�������£����е�������ӳ�����
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(j);
				break;
			case '4':// ���̡�4�������£����е�������ӳ�����
				g_map_x.at<float>(j, i) = static_cast<float>(g_srcImage.cols - i);
				g_map_y.at<float>(j, i) = static_cast<float>(g_srcImage.rows - j);
				break;
			}
		}
	}
	return 1;
}

//-----------------------------------��ShowHelpText( )������----------------------------------  
//      ���������һЩ������Ϣ  
//----------------------------------------------------------------------------------------------  
static void ShowHelpText()
{
	//���һЩ������Ϣ  
	printf("\n\n\n\t��ӭ������ӳ��ʾ������~\n\n");
	printf("\t��ǰʹ�õ�OpenCV�汾Ϊ OpenCV ");
	printf("\n\n\t��������˵��: \n\n"
		"\t\t���̰�����ESC��- �˳�����\n"
		"\t\t���̰�����1��-  ��һ��ӳ�䷽ʽ\n"
		"\t\t���̰�����2��- �ڶ���ӳ�䷽ʽ\n"
		"\t\t���̰�����3��- ������ӳ�䷽ʽ\n"
		"\t\t���̰�����4��- ������ӳ�䷽ʽ\n"
		"\n\n\t\t\t\t\t\t\t\t byǳī\n\n\n"
	);
}

int main_10()
{
	//�ı�console������ɫ
	system("color 3F");

	ShowHelpText();

	//����ԭʼͼ��Mat��������   
	Mat g_srcImage = imread("girl.png");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ

	//��ʾԭʼͼ  
	imshow("��ԭʼͼ��", g_srcImage);

	//����������
	namedWindow("��Ч��ͼ��", 1);
	createTrackbar("ֵ", "��Ч��ͼ��", &g_nthreshold, 200, on_HoughLines);

	//���б�Ե����ת��Ϊ�Ҷ�ͼ
	Canny(g_srcImage, g_midImage, 50, 200, 3);//����һ��canny��Ե���
	cvtColor(g_midImage, g_dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ

	//����һ�λص�����������һ��HoughLinesP����
	on_HoughLines(g_nthreshold, 0);
	HoughLinesP(g_midImage, g_lines, 1, CV_PI / 180, 80, 50, 10);

	//��ʾЧ��ͼ  
	imshow("��Ч��ͼ��", g_dstImage);


	waitKey(0);

	return 0;
}
#define WINDOW_NAME3 "������Warp��Rotate���ͼ��" 
//#include<opencv2/legacy/legacy.hpp>
//-----------------------------------��main( )������--------------------------------------------
//	����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-----------------------------------------------------------------------------------------------
int main_11()
{
	//��0�������Ķ���
	Mat src, src_gray, dst, abs_dst;

	//��1������ԭʼͼ  
	src = imread("girl.png");  //����Ŀ¼��Ӧ����һ����Ϊ1.jpg���ز�ͼ

	//��2����ʾԭʼͼ 
	imshow("��ԭʼͼ��ͼ��Laplace�任", src);

	//��3��ʹ�ø�˹�˲���������
	//GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//��4��ת��Ϊ�Ҷ�ͼ
	//cvtColor(src, src_gray, CV_RGB2GRAY);

	//��5��ʹ��Laplace����
	Laplacian(src, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);

	//��6���������ֵ���������ת����8λ
	convertScaleAbs(dst, abs_dst);

	//��7����ʾЧ��ͼ
	imshow("��Ч��ͼ��ͼ��Laplace�任", abs_dst);

	waitKey(0);

	//return 0;
	//��0���ı�console������ɫ
	system("color 1A");

	//��0����ʾ��ӭ�Ͱ�������
	ShowHelpText();

	//��1�������ز�ͼ
	Mat srcImage1 = imread("girl.png", 1);
	Mat srcImage2 = imread("book.png", 1);
	if (!srcImage1.data || !srcImage2.data)
	{
		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ����ͼƬ����~�� \n"); return false;
	}

	//��2��ʹ��SURF���Ӽ��ؼ���
	int minHessian = 700;//SURF�㷨�е�hessian��ֵ
	Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(minHessian);//����һ��SurfFeatureDetector��SURF�� ������������  
	std::vector<KeyPoint> keyPoint1, keyPoints2;//vectorģ���࣬����������͵Ķ�̬����

	//��3������detect��������SURF�����ؼ��㣬������vector������
	detector->detect(srcImage1, keyPoint1);
	detector->detect(srcImage2, keyPoints2);

	//��4������������������������
	Mat descriptors1, descriptors2;
	cv::Ptr<SURF> extractor = SURF::create();

	//SurfDescriptorExtractor extractor;
	
	extractor->compute(srcImage1, keyPoint1, descriptors1);
	extractor->compute(srcImage2, keyPoints2, descriptors2);

	//��5��ʹ��BruteForce����ƥ��
	// ʵ����һ��ƥ����
	BFMatcher matcher(NORM_L2);
	//BruteForceMatcher<L2<float>> matcher;
	std::vector< DMatch > matches;
	//ƥ������ͼ�е������ӣ�descriptors��
	matcher.match(descriptors1, descriptors2, matches);

	//��6�����ƴ�����ͼ����ƥ����Ĺؼ���
	Mat imgMatches;
	drawMatches(srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches);//���л���

	//��7����ʾЧ��ͼ
	imshow("ƥ��ͼ", imgMatches);

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

 
