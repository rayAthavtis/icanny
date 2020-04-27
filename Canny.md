<h1>目录</h1>

[toc]

# 1. 实验概述



​		Canny边缘检测算子是John F. Canny于 1986 年开发出来的一个多级边缘检测算法。



##  1.1 实验目的

		1. 学习边缘检测相关知识，了解其应用；
  		2. 学习Canny算法的基本原理并实现Canny边缘检测；
  		3. 了解OpenCV相关内容。



## 1.2 实验内容

​		不直接调用OpenCV的Canny函数实现，编写代码实现Canny边缘检测。



# 2. 实验步骤



## 2.1 平滑处理

​		做平滑处理的主要目的是消除噪声，因为未经处理的原始图像数据的噪声会影响边缘检测的效果，因此，Canny的第一步是将原始图像数据做高斯卷积，减小噪声对其的影响。



### 2.1.1 二维高斯



​		较小的sigma值产生的模糊效果较弱，这样就可以检测较小、变化明显的细线。较大的sigma值产生的模糊效果较强，这样就可以检测较大、较平滑的边缘。

```c++
// 高斯模糊：消除噪点
void gaussblur(Mat &src, Mat &dst, Size ksize, double sigma) {
    if( ksize.width == 1 && ksize.height == 1 ) {
        src.copyTo(dst);  //如果滤波器核的大小为1则用不着滤波
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
    // 一维高斯中做了边缘处理，这里没做
    for (int i=n/2; i<src.rows-n/2; i++)
    for (int j=n/2; j<src.cols-n/2; j++) {
        for (int p=0; p<n; p++)
        for (int q=0; q<n; q++)  // gauss核对称
            tmp.ptr(i)[j] += gauss[p][q] * src.ptr(i-n/2+p)[j-n/2+q];
    }
    tmp.copyTo(dst);
}
```



### 2.1.2 优化：一维高斯

​		对图像在两个独立的一维空间分别进行计算。水平方向进行一次模糊，在竖直方向进行一次模糊。与二维卷积空间处理的效果相同，但是时间复杂度大大降低。

```c++
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
```



* 图像处理结果



## 2.2 边缘强度计算

​		

### 2.2.1 sobel算子

​		Canny算法的基本思想是找寻一幅图像中灰度强度变化最强的位置。所谓变化最强，即指梯度方向。平滑后的图像中每个像素点的梯度可以由Sobel算子（一种卷积运算）来获得（opencv中有封装好的函数，可以求图像中每个像素点的n阶导数）。首先，利用如下的核来分别求得沿水平（x）和垂直（y）方向的梯度G~X~和G~Y~。

```c++
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
```



* 图像处理结果



## 2.3 非极大值抑制

这一步的目的是将模糊（blurred）的边界变得清晰（sharp）。通俗的讲，就是保留了每个像素点上梯度强度的极大值，而删掉其他的值。对于每个像素点，进行如下操作：
a) 将其梯度方向近似为以下值中的一个（0,45,90,135,180,225,270,315）（即上下左右和45度方向）
b) 比较该像素点，和其梯度方向正负方向的像素点的梯度强度
c) 如果该像素点梯度强度最大则保留，否则抑制（删除，即置为0）

```c++
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
```



* 图像处理结果



## 2.4 双阈值算法检测		

经过非极大抑制后图像中仍然有很多噪声点。Canny算法中应用了一种叫双阈值的技术。即设定一个阈值上界和阈值下界（opencv中通常由人为指定的），图像中的像素点如果大于阈值上界则认为必然是边界（称为强边界，strong edge），小于阈值下界则认为必然不是边界，两者之间的则认为是候选项（称为弱边界，weak edge），需进行进一步处理。双阈值：使用两个阈值比使用一个阈值更加灵活，但是它还是有阈值存在的共性问题。设置的阈值过高，可能会漏掉重要信息；阈值过低，将会把枝节信息看得很重要。很难给出一个适用于所有图像的通用阈值。还没有一个经过验证的实现方法。

```c++
// 双阈值算法
// Canny算法采用双阈值，高阈值一般是低阈值的两倍，遍历所有像素点
void dthresproc(Mat &src, Mat &dst, double low, double high) {
    src.copyTo(dst);
    for (int i=0; i<src.rows; i++)
    for (int j=0; j<src.cols; j++)
        if (dst.ptr(i)[j]>high) { dst.ptr(i)[j] = 255; }
        else if (dst.ptr(i)[j]<low) { dst.ptr(i)[j] = 0; }
    imshow("dtproc", dst);
    imwrite("outImgs/dtproc.jpg", dst);
}
```



* 图像处理结果



## 2.5 滞后阈值化处理

和强边界相连的弱边界认为是边界，其他的弱边界则被抑制。

```c++
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
```



* 图像处理结果



## 2.6 iCanny()函数



```c++
void icanny(Mat &src, Mat &dst, double low, double high) {
    Mat sob, nmax;
    int thta[(src.rows-1)*(src.cols-1)];

    sobel(src, sob, thta);
    imshow("sobel", sob);
    imwrite("outImgs/sobel.jpg", sob);

    Nmaxsup(sob, nmax, thta);
    imshow("nmax", nmax);
    imwrite("outImgs/nmax.jpg", nmax);

    dthresproc(nmax, dst, low, high);
    edgeconnect(dst);
}
```



## 2.6 效果对比

​		与OpenCV的Canny效果进行对比

```c++
Canny(gauss, canny, 50, 100);
```

* iCanny()效果
* Canny()效果



# 3. 实验结论和感想

​		对canny算法有了更加深入的理解。