#include "FeaturesContainer.h"

FeaturesContainer::FeaturesContainer()
{

}

FeaturesContainer::~FeaturesContainer()
{

}

void FeaturesContainer::appendFeatures(const vector<cv::Mat> &responses){
    for (cv::Mat response : responses){
        assert(response.channels()==1); // feature matrix must be 2D mat
        for(int row=0; row<response.rows; row++){
            const double *p=response.ptr<double>(row);
            for(int col=0; col<response.cols; col++){
                _allFeatures.back().push_back(p[col]);
            }
        }
    }
}

void FeaturesContainer::appendFeatures(const vector<cv::Mat> &responses, const cv::Mat &map){

    for (cv::Mat response : responses){
        assert(response.channels()==1); // feature matrix must be 2D mat
        for(int row=0; row<response.rows; row++){
            const double *p=response.ptr<double>(row);
            for(int col=0; col<response.cols; col++){
               if(map.at<int>(row, col)>0) _allFeatures.back().push_back(p[col]);
            }
        }
    }
}

void FeaturesContainer::setFeatures(const vector<cv::Mat> responses, const SuperPixelContainer &container){
//    _allFeatures.clear();
    _allFeatures.resize(container.size());
    for (int c = 0; c < responses.size(); c++) {
        for (int y = 0; y < container.height(); y++) {
            const int *segId = container[0].ptr<int>(y);
            const double *p = responses[c].ptr<const double>(y);
            for (int x = 0; x < container.width(); x++) {
                if (segId[x] < 0) continue;
                _allFeatures[segId[x]].push_back(p[x]);
            }
        }
    }
}
