#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <cstring>

using namespace cv;
using namespace ml;
using namespace std;

class DigitRecognizer
{
public:
    DigitRecognizer();

    ~DigitRecognizer();

    bool train(char* trainPath, char* labelsPath);

    int classify(Mat img);

private:
    Mat preprocessImage(Mat img);

    int readFlippedInteger(FILE *fp);

private:
    Ptr<ml::KNearest> knn;
	
    Mat train_data_mat, train_label_mat;
    int numRows, numCols, numImages;

};
