// for knn train use this file https://gist.github.com/johnhany/a48487dcacdb4c2108e919b421d19cfb

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include "digitrecognizer.h"
#include <bits/stdc++.h>

using namespace cv;
using namespace ml;
using namespace std;

typedef uint8_t BYTE;
#define PIXELS_IN_IMAGE 28*28

DigitRecognizer::DigitRecognizer()
{
    knn = KNearest::create();
}

DigitRecognizer::~DigitRecognizer()
{
    delete knn;
}

int DigitRecognizer::readFlippedInteger(FILE *fp)
{
    int ret = 0;

    BYTE *temp;

    temp = (BYTE *)(&ret);
    fread(&temp[3], sizeof(BYTE), 1, fp);
    fread(&temp[2], sizeof(BYTE), 1, fp);
    fread(&temp[1], sizeof(BYTE), 1, fp);

    fread(&temp[0], sizeof(BYTE), 1, fp);

    return ret;
}

bool DigitRecognizer::train(char* trainPath, char* labelsPath)
{
    FILE *fp = fopen(trainPath, "rb");
    FILE *fp2 = fopen(labelsPath, "rb");

    if (!fp || !fp2)
    {
        cout << "can't open file" << endl;
    }

    int magicNumber = readFlippedInteger(fp);
    numImages = readFlippedInteger(fp);
    numRows = readFlippedInteger(fp);
    numCols = readFlippedInteger(fp);
    fseek(fp2, 0x08, SEEK_SET);

    int size = numRows * numCols;

    cout << "size: " << size << endl;
    cout << "rows: " << numRows << endl;
    cout << "cols: " << numCols << endl;

    Mat_<float> trainFeatures(numImages, size);
    Mat_<int> trainLabels(1, numImages);

    BYTE *temp = new BYTE[size];
    BYTE tempClass = 0;
    for (int i = 0; i < numImages; i++)
    {
        fread((void *)temp, size, 1, fp);
        fread((void *)(&tempClass), sizeof(BYTE), 1, fp2);

        trainLabels[0][i] = (int)tempClass;

        for (int k = 0; k < size; k++)
        {
            trainFeatures[i][k] = (float)temp[k];
        }
    }
    knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
    fclose(fp);
    fclose(fp2);
	return true;
}

int DigitRecognizer::classify(cv::Mat img)
{
    static int counter = 0;
    counter++;
    Mat cloneImg = preprocessImage(img);
    // cloneImg.convertTo(cloneImg, CV_32FC1);
    cloneImg.reshape(1,1);
    cloneImg.convertTo(cloneImg, CV_32F);
    
    // cv::imshow("img" + to_string(counter), cloneImg);
    Mat response, dist;
    return knn->findNearest(cloneImg, 1, response);
}

Mat DigitRecognizer::preprocessImage(Mat img)
{
    int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

    Mat temp;
    int thresholdBottom = 50;
    int thresholdTop = 50;
    int thresholdLeft = 50;
    int thresholdRight = 50;
    int center = img.rows/2;
    for(int i=center;i<img.rows;i++)
    {
        if(rowBottom==-1)
        {
            temp = img.row(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdBottom || i==img.rows-1)
                rowBottom = i;

        }

        if(rowTop==-1)
        {
            temp = img.row(img.rows-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdTop || i==img.rows-1)
                rowTop = img.rows-i;

        }

        if(colRight==-1)
        {
            temp = img.col(i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdRight|| i==img.cols-1)
                colRight = i;

        }

        if(colLeft==-1)
        {
            temp = img.col(img.cols-i);
            IplImage stub = temp;
            if(cvSum(&stub).val[0] < thresholdLeft|| i==img.cols-1)
                colLeft = img.cols-i;
        }
    }
    Mat newImg;

    newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

    int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
    {
        uchar *ptr = newImg.ptr<uchar>(y);
        for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
        {
            ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
        }
    }
    Mat cloneImg = Mat(numRows, numCols, CV_8UC1);

    resize(newImg, cloneImg, Size(numCols, numRows));

    // Now fill along the borders
    for(int i=0;i<cloneImg.rows;i++)
    {
        floodFill(cloneImg, cvPoint(0, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(cloneImg.cols-1, i), cvScalar(0,0,0));

        floodFill(cloneImg, cvPoint(i, 0), cvScalar(0));
        floodFill(cloneImg, cvPoint(i, cloneImg.rows-1), cvScalar(0));
    }
     cloneImg = cloneImg.reshape(1, 1);

    return cloneImg;
    // Moments m = moments(img);
    // if(abs(m.mu02) < 1e-2)
    // {
    //     // No deskewing needed.
    //     return img.clone();
    // }
    // // Calculate skew based on central momemts.
    // double skew = m.mu11/m.mu02;
    // // Calculate affine transform to correct skewness.
    // Mat warpMat = (Mat_<double>(2,3) << 1, skew, -0.5*2*skew, 0, 1 , 0);
     
    // Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    // warpAffine(img, imgOut, warpMat, imgOut.size());

    // return imgOut;
}
