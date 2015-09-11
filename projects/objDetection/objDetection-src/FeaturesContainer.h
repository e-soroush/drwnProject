#ifndef FEATURESCONTAINER_H
#define FEATURESCONTAINER_H
#include "opencv2/opencv.hpp"
#include "SuperPixelContainer.h"

using namespace std;

class FeaturesContainer
{
public:
    FeaturesContainer();
    ~FeaturesContainer();
    void appendFeatures(const vector<cv::Mat> &responses);
    void appendFeatures(const vector<double> &features);
    void appendFeatures(const vector<cv::Mat> &responses, const cv::Mat &map);
    int getNumFeatures(){return _numFeatures;}
    void setFeatures(const vector<cv::Mat> responses, const SuperPixelContainer &container);


protected:
    vector<vector<double> >_allFeatures;
    int _numFeatures;
};

#endif // FEATURESCONTAINER_H
