#ifndef HOGFEATURES_H
#define HOGFEATURES_H
#include "drwnVision.h"
#include "SuperPixelContainer.h"

class HOGFeatures : public drwnHOGFeatures
{
public:
    HOGFeatures();
    ~HOGFeatures();
    void computeFeatures(const cv::Mat &img, std::vector<std::vector<double> > &features, SuperPixelContainer &container);
};

#endif // HOGFEATURES_H
