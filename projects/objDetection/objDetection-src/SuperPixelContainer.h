#ifndef SUPERPIXELCONTAINER_H
#define SUPERPIXELCONTAINER_H
#include "drwnVision.h"
#include "opencv2/opencv.hpp"


class SuperPixelContainer : public drwnSuperpixelContainer
{
public:
    /// default constructor
    SuperPixelContainer();
    ~SuperPixelContainer();
    /// select given super pixel and highlight it
    cv::Mat selectSuperPixel(unsigned segId, const cv::Mat& img, vector<cv::Mat> &selected);
    void setMap(vector<cv::Mat> &selected){_maps=selected;}
    void getMap(vector<cv::Mat> &selected){selected=_maps;}
    int getSuperPixelId(int x, int y);
    void gTruthSuperPixel(vector<vector<int> > &segIds, const MatrixXi &labels);
    cv::Mat visualize(const cv::Mat& img, bool bColorById, bool gTruth=false) const;


};

#endif // SUPERPIXELCONTAINER_Hselected
