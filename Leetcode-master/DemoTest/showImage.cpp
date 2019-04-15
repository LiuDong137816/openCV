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

//-----------------------------------��ShowHelpText( )������----------------------------------
//		���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//���һЩ������Ϣ
	printf("\n\n\n\t������������۲�ͼ��Ч��~\n\n");
	printf("\n\n\t��������˵��: \n\n"
		"\t\t���̰�����ESC�����ߡ�Q��- �˳�����\n"
		"\t\t���̰�����1��- ʹ����Բ(Elliptic)�ṹԪ��\n"
		"\t\t���̰�����2��- ʹ�þ���(Rectangle )�ṹԪ��\n"
		"\t\t���̰�����3��- ʹ��ʮ����(Cross-shaped)�ṹԪ��\n"
		"\t\t���̰������ո�SPACE��- �ھ��Ρ���Բ��ʮ���νṹԪ����ѭ��\n"
		"\n\n\t\t\t\t\t\t\t\t byǳī"
	);
}

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

void resize_pyr() {
	char ch;
	Mat g_tmpImage;
	g_srcImage = imread("girl.png");//����Ŀ¼����Ҫ��һ����Ϊ1.jpg�Ĳ���ͼ������ߴ��豻2��N�η�������NΪ�������ŵĴ���
	if (!g_srcImage.data) { printf("Oh��no����ȡsrcImage����~�� \n"); return; }

	// ������ʾ����
	g_tmpImage = g_srcImage;
	namedWindow("ͼƬ", CV_WINDOW_AUTOSIZE);
	imshow("ͼƬ", g_srcImage);
	while (cin >> ch)
	{
		switch (ch)
		{
		case '1':
			pyrUp(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">��⵽������A�������£���ʼ���л��ڡ�pyrUp��������ͼƬ�Ŵ�ͼƬ�ߴ��2 \n");
			break;
		case '2':
			resize(g_tmpImage, g_dstImage, Size(g_tmpImage.cols * 2, g_tmpImage.rows * 2));
			printf(">��⵽������W�������£���ʼ���л��ڡ�resize��������ͼƬ�Ŵ�ͼƬ�ߴ��2 \n");
			break;
			
		}
	}
}

//-----------------------------------��main( )������--------------------------------------------
//	����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-----------------------------------------------------------------------------------------------
int main()
{
	//�ı�console������ɫ
	system("color 2F");

	//��ʾ��ӭ��
	ShowHelpText();

	//����ԭͼ
	g_srcImage = imread("girl.png");
	if (!g_srcImage.data) { printf("Oh��no����ȡsrcImage����~�� \n"); return false; }

	//��ʾԭʼͼ
	namedWindow("��ԭʼͼ��");
	imshow("��ԭʼͼ��", g_srcImage);

	// ������srcͬ���ͺʹ�С�ľ���(dst)
	g_dstImage.create(g_srcImage.size(), g_srcImage.type());

	// ��ԭͼ��ת��Ϊ�Ҷ�ͼ��
	cvtColor(g_srcImage, g_srcGrayImage, CV_BGR2GRAY);

	// ������ʾ����
	namedWindow("��Ч��ͼ��Canny��Ե���", CV_WINDOW_AUTOSIZE);
	namedWindow("��Ч��ͼ��Sobel��Ե���", CV_WINDOW_AUTOSIZE);

	// ����trackbar
	createTrackbar("����ֵ��", "��Ч��ͼ��Canny��Ե���", &g_cannyLowThreshold, 120, on_Canny);
	createTrackbar("����ֵ��", "��Ч��ͼ��Sobel��Ե���", &g_sobelKernelSize, 3, on_Sobel);

	// ���ûص�����
	on_Canny(0, 0);
	on_Sobel(0, 0);

	//���÷�װ��Scharr��Ե������ĺ���
	Scharr();

	//��ѯ��ȡ������Ϣ��������Q�������˳�
	while ((char(waitKey(1)) != 'q')) {}

	return 0;
}

 
