//
//  main.cpp
//  icanny
//
//  Created by 任蕾 on 2020/3/30.
//  Copyright © 2020 任蕾. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>

using namespace cv;
using namespace std;

/*
 Canny边缘检测子算法
 1.平滑处理：将图像与尺度为σ的高斯函数做卷积。
 2.边缘强度计算：计算图像的边缘幅度及方向。
 3.非极大值抑制：只有局部极大值标记为边缘。
 4.滞后阈值化处理。
 
 5.对于递增的σ，重复步骤1~4。
 6.用特征综合方法，搜集多尺度的最终边缘信息。
 一般做前四步
 
 一、用高斯滤波器平滑图像
 二、用Sobel等梯度算子计算梯度幅值和方向
 三、对梯度幅值进行非极大值抑制
 四、用双阈值算法检测和连接边缘
 
*/

#define PI 3.1415926

// 高斯模糊：消除噪点
void gaussblur(Mat &src, Mat &dst, Size ksize, double sigma) {
    if( ksize.width == 1 && ksize.height == 1 ) {
        src.copyTo(dst);//如果滤波器核的大小为1则用不着滤波
        return ;
    }
    int n = ksize.width;
    double gauss[n][n];  // n*n数组
    double sum = 0;
    // 二维高斯
    for (int i=0; i<n; i++)
    for (int j=0; j<n; j++) {
        int x = i - n/2;
        int y = j - n/2;
        gauss[i][j] = exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)));  // 系数可忽略（归一化）
        sum += gauss[i][j];
    }
    // 归一化
    for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
        gauss[i][j] /= sum;

    Mat tmp(src.rows, src.cols, CV_8U, Scalar(0));
    // 无边缘处理
    for (int i=n/2; i<src.rows-n/2; i++)
    for (int j=n/2; j<src.cols-n/2; j++) {
        for (int p=0; p<n; p++)
        for (int q=0; q<n; q++)  // gauss核对称
            tmp.ptr(i)[j] += gauss[p][q] * src.ptr(i-n/2+p)[j-n/2+q];
    }
    tmp.copyTo(dst);
}

/*
优化：
对图像在两个独立的一维空间分别进行计算。
水平方向进行一次模糊，在竖直方向进行一次模糊。
与二维卷积空间处理的效果相同
时间复杂度down
 */

void gaussblur1D(Mat &src, Mat &dst, Size ksize, double sigma) {
    if( ksize.width == 1 && ksize.height == 1 ) {
        src.copyTo(dst);//如果滤波器核的大小为1则用不着滤波
        return ;
    }
    int n = ksize.width;
    double gauss[n];
    double sum = 0;
    // 一维高斯
    for (int i=0; i<n; i++) {
        int x = i - n/2;
        gauss[i] = exp(-pow(x, 2) / (2 * pow(sigma, 2)));
        sum += gauss[i];
    }
    // 归一化
    for (int i=0; i<n; i++)
        gauss[i] /= sum;
    
    Mat tmp(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i=0; i<src.rows; i++)
    for (int j=0; j<src.cols; j++)
        for (int p=0; p<n; p++) {
            int x = i-n/2+p;
            if (x>src.rows-1) { x = 2 * (src.rows-1) - x; }
            else if (x<0) { x = -x; }
            tmp.ptr(i)[j] += gauss[p] * src.ptr(x)[j];
        }
    // imshow("gauss0", tmp);
    // 换向
    Mat tmp2(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i=0; i<src.rows; i++)
    for (int j=0; j<src.cols; j++)
        for (int p=0; p<n; p++) {
            int y = j-n/2+p;
            if (y>src.cols-1) { y = 2 * (src.cols-1) - y; }
            else if (y<0) { y = -y; }
            tmp2.ptr(i)[j] += gauss[p] * tmp.ptr(i)[y];
    }
    tmp2.copyTo(dst);
}

// sobel算子：找边缘
void sobel(Mat &src, Mat &dst, int theta[]) {
    Mat tmp(src.rows, src.cols, CV_8U, Scalar(0));
    int k = 0;
    // cout<<typeid(src.ptr(1)[1]).name()<<endl;
    // val: 159
    for (int i=1; i<src.rows-1; i++)
    for (int j=1; j<src.cols-1; j++) {
        // x方向 梯度方向朝右为正
        int x = src.ptr(i-1)[j+1] + 2 * src.ptr(i)[j+1] + src.ptr(i+1)[j+1]
        - src.ptr(i-1)[j-1] - 2 * src.ptr(i)[j-1] - src.ptr(i+1)[j-1];
        // y方向 梯度方向朝上为正
        int y = src.ptr(i-1)[j-1] + 2 * src.ptr(i-1)[j] + src.ptr(i-1)[j+1]
        - src.ptr(i+1)[j-1] - 2 * src.ptr(i+1)[j] - src.ptr(i+1)[j+1];
        // a11  三维算法计算梯度
        // X = a02 + 2 * a12 + a22 - a00 - 2 * a10 - a20;
        // Y = a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22;
        
        int sum = sqrt(pow(x, 2) + pow(y, 2));  // 求平方根
        // int sum = abs(x) + abs(y);  // 求绝对值
        sum = sum > 255 ? 255 : sum;
        sum = sum < 0 ? 0 : sum;
        tmp.ptr(i)[j] = sum;  // 指针，效率相对at较高
        // 使用0代表(3*PI/8, PI/2]V[-PI/2, -3*PI/8)，1代表[-3*8/PI, -PI/8]，2代表[-PI/8, PI/8)，3代表[PI/8, 3*PI/8]
        // 处理特殊情况x=0||y=0
        if (y==0)
            theta[k] = 2;
        else if (x==0)
            theta[k] = 0;
        else {
            int tmp = atan(y/x);
            if (abs(tmp)>3*PI/8) { theta[k] = 0; }
            else if (tmp<-PI/8) { theta[k] = 1; }
            else if (tmp<PI/8) { theta[k] = 2; }
            else {theta[k] = 3; }
        }
        k++;
    }
    tmp.copyTo(dst);
}

// 非极大值抑制：消除噪点，细化边缘
void Nmaxsup(Mat &src, Mat &dst, int theta[]) {
    int k = 0;
    src.copyTo(dst);  // clone()会申请新地址
    for (int i=1; i<src.rows-1; i++)
    for (int j=1; j<src.cols-1; j++) {
        if (dst.ptr(i)[j]==0) { k++; continue; }
        int a = 0, b = 0;
        if (theta[k]==0) { a = 1; }
        else if (theta[k]==1) { a = 1; b = 1; }
        else if (theta[k]==2) { b = 1; }
        else { a = 1; b = -1; }
        // 沿着该点梯度方向，若该点小于前后两点幅值，则置0
        if (dst.ptr(i)[j]<=dst.ptr(i-a)[j-b] || dst.ptr(i)[j]<=dst.ptr(i+a)[j+b])
            dst.ptr(i)[j] = 0;
        k++;
    }
}

// 双阈值算法
// Canny算法采用双阈值，高阈值一般是低阈值的两倍，遍历所有像素点
void dthresproc(Mat &src, Mat &dst, double low, double high) {
    src.copyTo(dst);
    for (int i=0; i<src.rows; i++)
    for (int j=0; j<src.cols; j++)
        if (dst.ptr(i)[j]>high) { dst.ptr(i)[j] = 255; }
        else if (dst.ptr(i)[j]<low) { dst.ptr(i)[j] = 0; }
    imshow("dtproc", dst);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/dtproc.jpg", dst);
}

// 连接
// 强边缘点的8邻点域的弱边缘点置为255
void edgeconnect(Mat &src) {
    for (int i=1; i<src.rows-1; i++)
    for (int j=1; j<src.cols-1; j++)
        if (src.ptr(i)[j]==255)
            for (int m=-1; m<2; m++)
            for (int n=-1; n<2; n++) {
                if (src.ptr(i+m)[j+n]!=0  // 非抑制点
                && src.ptr(i+m)[j+n]!=255  // 非强边缘点
                )
                    src.ptr(i+m)[j+n] = 255;
            }
    for (int i=1; i<src.rows-1; i++)
    for (int j=1; j<src.cols-1; j++)
        if (src.ptr(i)[j]!=255)  // 孤立弱边缘点
            src.ptr(i)[j] = 0;
}

void icanny(Mat &src, Mat &dst, double low, double high) {
    Mat sob, nmax;
    int thta[(src.rows-1)*(src.cols-1)];
    
    sobel(src, sob, thta);
    imshow("sobel", sob);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/sobel.jpg", sob);

    Nmaxsup(sob, nmax, thta);
    imshow("nmax", nmax);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/nmax.jpg", nmax);
    
    dthresproc(nmax, dst, low, high);
    edgeconnect(dst);
}

int main() {
    Mat img = imread("/Users/renlei/XcodeSpace/icanny/imgs/lena.jpg", IMREAD_GRAYSCALE);
    if (!img.data) {
        cout<<"img fault!"<<endl;
        return 0;
    }
    imshow("src", img);
    Mat gauss, canny, icy;
    // cvtColor(img, gray, CV_BGR2GRAY);
    // imshow("gray", gray);
    // blur(grayImg, edge, Size(3, 3));
    // GaussianBlur(gray, gauss, Size(3, 3), 3);
    // gaussblur(img, gauss, Size(3, 3), 3);
    gaussblur1D(img, gauss, Size(3, 3), 3);
    imshow("gauss", gauss);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/gauss.jpg", gauss);

    Canny(gauss, canny, 50, 100);
    imshow("canny", canny);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/canny.jpg", canny);

    icanny(gauss, icy, 50, 100);
    imshow("icanny", icy);
    imwrite("/Users/renlei/XcodeSpace/icanny/outImgs/icanny.jpg", icy);
    waitKey(0);
    return 0;
}
