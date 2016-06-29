//
//  Imagesmodule.cpp
//  ImageModule
//
//  Created by setomasakazu on 2015/06/27.
//  Copyright (c) 2015年 setomasakazu. All rights reserved.
//

#include "Imagesmodule.h"

#ifdef __ANDROID__
    #include <jni.h>
    #include <opencv/cv.h>
    #include <iostream>
//    #include <opencv2/imgproc.hpp>
//    #include <opencv2/highgui.hpp>
//    #include <opencv2/contrib/contrib.hpp>
    #include "opencv2/opencv.hpp"
    #include "opencv2/superres/optical_flow.hpp"
    #include <GLES/gl.h>
    #include <GLES/glext.h>
    #include <GLES2/gl2.h>
#else
    #import "opencv2/opencv.hpp"
    #include <OpenGLES/ES2/gl.h>
    #include <OpenGLES/ES2/glext.h>
#endif

using namespace cv;
using namespace std;

int SetTextureOfCam1(unsigned char* src, int width,  int height)
{
#ifdef __ANDROID__
    cv::Mat gray;
    cv::Mat argb;
    
    cv::Mat frame(height, width, CV_8UC4, src);
    // RGBA -> グレースケール画像
    cvtColor(frame, gray, CV_RGBA2GRAY);
    cv::cvtColor(gray, argb, CV_GRAY2RGBA);
    std::memcpy(src, argb.data, argb.total() * argb.elemSize());
    return 0;
#else
    cv::Mat gray;
    cv::Mat argb;
    
    cv::Mat frame(height, width, CV_8UC4, src);
    // RGBA -> グレースケール画像
    cvtColor(frame, gray, CV_RGBA2GRAY);
    cv::cvtColor(gray, argb, CV_GRAY2RGBA);
    std::memcpy(src, argb.data, argb.total() * argb.elemSize());
    return 0;
#endif
}

void drawOptFlowMap(const cv::Mat& flow, cv::Mat& base, int step,
                    double, const cv::Scalar& color)
{
    for(int y = 0; y < base.rows; y += step)
        for(int x = 0; x < base.cols; x += step)
        {
            const cv::Point2f& u = flow.at<cv::Point2f>(y, x);
            cv::line(base, cv::Point(x,y), cv::Point(cvRound(x+u.x), cvRound(y+u.y)),
                     color);
            cv::circle(base, cv::Point(x,y), 2, color, -1);
        }
}

//モーション追跡
void OpticalFlow1(unsigned char* src, unsigned char* pre, int width, int height)
{
    cv::Mat red_img(height, width, CV_8UC3, cv::Scalar(0,0,255));
    
    cv::Mat frame(height, width, CV_8UC4, src);
    cv::Mat gray;
    cv::Mat prevgray(height, width, CV_8UC1, pre);
    cv::Mat flow;
    cv::Mat back;
    
    // RGBA -> グレースケール画像
    cvtColor(frame, gray, CV_RGBA2GRAY);
    if( !prevgray.empty() )
    {
        
        cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cvtColor(prevgray, back, CV_GRAY2RGBA);
        drawOptFlowMap(flow, back, 16, 1.5, cv::Scalar(0, 255, 0));
    }else{
        cvtColor(gray, back, CV_GRAY2RGBA);
    }
    std::memcpy(pre, gray.data, gray.total() * gray.elemSize());
    std::memcpy(src, back.data, back.total() * back.elemSize());
}

//モーション追跡
void OpticalFlow2(unsigned char* src, int width, int height)
{
    IplImage* prevgray = 0;
    // RGBA source : frame
    IplImage* frame = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 4);
    cvSetData(frame, src, frame->widthStep);
    // グレースケール画像
    IplImage* gray = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    // RGBA -> グレースケール画像
    cvCvtColor(frame, gray, CV_RGBA2GRAY);
    if(prevgray != 0) {
        
        
    }
    std::swap(prevgray, gray);
    // グレースケール画像 -> RGBA
    cvCvtColor(prevgray, frame, CV_GRAY2RGBA);
    cvReleaseImageHeader(&frame);
    cvReleaseImage(&gray);
}

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;
    
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];
    
    if (first)
    {
        int k = 0;
        
        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);
        
        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);
        
        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);
        
        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);
        
        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);
        
        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);
        
        first = false;
    }
    
    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;
    
    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;
    
    Vec3b pix;
    
    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.f;
        const float col1 = colorWheel[k1][b] / 255.f;
        
        float col = (1 - f) * col0 + f * col1;
        
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        
        pix[2 - b] = static_cast<uchar>(255.f * col);
    }
    
    return pix;
}

static void drawOpticalFlow3(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1)
{
    dst.create(flow.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));
    
    // determine motion range:
    float maxrad = maxmotion;
    
    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);
                
                if (!isFlowCorrect(u))
                    continue;
                
                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }
    
    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);
            
            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

//オプティカルフローを可視化する。
//縦横のベクトルの強さを色に変換する。
//左：赤、右：緑、上：青、下：黄色
void visualizeFarnebackFlow(
                            const Mat& flow,    //オプティカルフロー CV_32FC2
                            Mat& visual_flow    //可視化された画像 CV_32FC3
)
{
    visual_flow = Mat::zeros(flow.rows, flow.cols, CV_32FC3);
    int flow_ch = flow.channels();
    int vis_ch = visual_flow.channels();//3のはず
    for(int y = 0; y < flow.rows; y++) {
        float* psrc = (float*)(flow.data + flow.step * y);
        float* pdst = (float*)(visual_flow.data + visual_flow.step * y);
        for(int x = 0; x < flow.cols; x++) {
            float dx = psrc[0];
            float dy = psrc[1];
            float r = (dx < 0.0) ? abs(dx) : 0;
            float g = (dx > 0.0) ? dx : 0;
            float b = (dy < 0.0) ? abs(dy) : 0;
            r += (dy > 0.0) ? dy : 0;
            g += (dy > 0.0) ? dy : 0;
            
            pdst[0] = b;
            pdst[1] = g;
            pdst[2] = r;
            
            psrc += flow_ch;
            pdst += vis_ch;
        }
    }
}

//モーション追跡
void OpticalFlow3(unsigned char* src, unsigned char* pre, int width, int height)
{
    cv::Mat frame(height, width, CV_8UC4, src);
    cv::Mat gray;
    cv::Mat frame_pre(height, width, CV_8UC4, pre);
    cv::Mat prevgray;
    cv::Mat flow;
    cv::Mat visual_flow;
    cv::Mat magnitude;
    cv::Mat angle;
    cv::Mat back;
    
    // RGBA -> グレースケール画像
    cvtColor(frame, gray, CV_RGBA2GRAY);
    cvtColor(frame_pre, prevgray, CV_RGBA2GRAY);
    if( !prevgray.empty() )
    {
/*
        cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);
        
        Mat xy[2]; //X,Y
        split(flow, xy);
        
        // 2次元ベクトルの角度と大きさを求めます
        // パラメタ:
        // x – x 座標の配列．必ず，単精度または倍精度の浮動小数点型配列です．
        // y – x と同じサイズ，同じ型の y 座標の配列．
        // magnitude – 大きさの出力配列． x と同じサイズ，同じ型です．
        // angle – 角度の出力配列． x と同じサイズ，同じ型．角度はラジアン ( 0 から  2 \pi ) あるいは度 (0 から 360) で表されます．
        // angleInDegrees – 角度の表記にラジアン（デフォルト）または度のどちらを用いるかを指定するフラグ．
        xy[0] = xy[0] * 10e6;
        xy[1] = xy[1] * 10e6;
        cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
        //        std::cout << "angle1" << std::endl << angle << std::endl << std::endl;
        //        std::cout << "magnitude1" << std::endl << magnitude << std::endl << std::endl;
        
        // 速度ベクトルのX軸と為す角度 -> 色相
        // 速度の大きさ -> 輝度
        cv::Mat hasvPlanes[3];
        normalize(angle, angle, 0, 1, NORM_MINMAX);
        hasvPlanes[0] = angle * 360;
        normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
        hasvPlanes[1] = magnitude * 2550;
        hasvPlanes[2] = Mat::ones(magnitude.size(), CV_32F)*100;
        cv::Mat hsv;
        cv::merge(hasvPlanes, 3, hsv);
        
        int img_width = angle.cols;
        int img_height = angle.rows;
        
        int hbins = 360;
        int histSize[] = {hbins};
        float hranges[] = { 0, 360 };
        const float* ranges[] = { hranges };
        MatND hist;
        int channels[] = {0};
        calcHist( &hsv, 1, channels, Mat(), // マスクは利用しません
                 hist, 1, histSize, ranges,
                 true, // ヒストグラムは一様です
                 false );
        
        // histogramを描画するための画像領域を確保
        Mat histImg(cv::Size(img_width, img_height), CV_8UC3, Scalar::all(255));
        
        // ヒストグラムのスケーリング
        // ヒストグラムのbinの中で、頻度数最大のbinの高さが、ちょうど画像の縦幅と同じ値になるようにする
        double maxVal=0;
        double minVal=0;
        cv::Point min_pt;
        cv::Point max_pt;
        minMaxLoc(hist, &minVal, &maxVal, &min_pt, &max_pt);
        int angle = saturate_cast<int>(hsv.at<float>(max_pt.y));
        hist = hist * (maxVal ? img_height / maxVal : 0.0);
        
        // ヒストグラムのbinの数だけ矩形を書く
        for( int j = 0; j < hbins; j++ ) {
            // saturate_castは、安全に型変換するための関数。桁あふれを防止
            int bin_w = cv::saturate_cast<int>((double)img_width / hbins);
            cv::rectangle(
                          histImg,
                          cv::Point(j*bin_w, histImg.rows),
                          cv::Point((j + 1)*bin_w, histImg.rows - cv::saturate_cast<int>(hist.at<float>(j))),
                          cv::Scalar(255, 0, 0), -1);
        }
        stringstream ss;
        ss << angle;
        cv::putText(histImg, ss.str(),
                    cv::Point(100,100),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200));
        Mat back;
        Mat h_img;
        cv::flip(histImg, h_img,1);
        cvtColor(h_img, back, CV_HSV2RGB,4);
        
        std::memcpy(src, back.data, back.total() * back.elemSize());
*/
        cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cvtColor(prevgray, back, CV_GRAY2RGBA);
        drawOptFlowMap(flow, back, 16, 1.5, cv::Scalar(0, 255, 0));
        std::memcpy(src, back.data, back.total() * back.elemSize());
    }
}

// 画像からエッジ画像を検出
void UpdateTexture(char* data, int width, int height)
{
    // RGBA
    IplImage* p = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 4);
    cvSetData(p, data, p->widthStep);
    // グレースケール画像
    IplImage* g = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    // RGBA -> グレースケール画像
    cvCvtColor(p, g, CV_RGBA2GRAY);
    // Cannyエッジ検出
    cvCanny(g, g, 50.0, 200.0);
    // グレースケール画像 -> RGBA
    cvCvtColor(g, p, CV_GRAY2RGBA);
    cvReleaseImageHeader(&p);
    cvReleaseImage(&g);
}

// iro filter
void DetectHsv(unsigned char* src, unsigned char* dest, int width, int height)
{
    cv::Mat hsv_skin_img(height,width,CV_8UC3);
    cv::Mat smooth_img(height,width,CV_8UC4);
    cv::Mat hsv_img(height,width,CV_8UC4);
    cv::Mat result_img(height,width, CV_8UC4);
    
    cv::Mat input_img(height, width, CV_8UC4, src);

    cv::medianBlur(input_img,smooth_img,7);	//ノイズがあるので平滑化
    cv::cvtColor(smooth_img,hsv_img,CV_BGR2HLS);	//HSVに変換

    for(int y=0; y<height;y++)
    {
        for(int x=0; x<width; x++)
        {
            long a = hsv_img.step*y+(x*3);
            if(hsv_img.data[a] >=0 && hsv_img.data[a] <=15 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50 ) //HSVでの検出
            {
                hsv_skin_img.data[a] = 255; //肌色部分を青に
            }
        }
    }

    cv::cvtColor(hsv_img,result_img,CV_HSV2BGR);	//RGBに変換
    std::memcpy(dest, result_img.data, result_img.total() * result_img.elemSize());
    
}

void convertToCannyTextureA(char* src, int width, int height, float threshold1, float threshold2)
{
    cv::Mat hsv, mask, destImg;                          // 画像オブジェクトの作成

    cv::Mat aimg(height, width, CV_8UC4, src);
    cv::Mat srcImg(height, width, CV_8UC3, src);
    cvtColor(srcImg, hsv, CV_RGB2HSV);                   // 画像をRGBからHSVに変換
    // 色検出でマスク画像の作成
    cv::Scalar lowerb = cv::Scalar(150,70,70);           //下限値
    cv::Scalar upperb = cv::Scalar(360,255,255);         //上限値
    cv::inRange(hsv, lowerb, upperb, mask);
    
//    cv::Mat dst3;
//    cv::bitwise_and(srcImg, mask, dst3);

    
//    cv::cvtColor(dst3, destImg, CV_GRAY2RGB);
    std::memcpy(src, hsv.data, hsv.total() * hsv.elemSize());
}

void createCheckTexture(unsigned char* arr, int w, int h, int ch)
{
    int n = 0;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < ch; ++k) {
                arr[n++] = ( (i + j) % 2 == 0 ) ? 255 : 0;
            }
        }
    }
}

