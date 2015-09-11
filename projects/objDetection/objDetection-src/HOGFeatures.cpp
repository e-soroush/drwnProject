#include "HOGFeatures.h"

HOGFeatures::HOGFeatures()
{

}

HOGFeatures::~HOGFeatures()
{

}

void HOGFeatures::computeFeatures(const Mat &img, std::vector<std::vector<double> > &features, SuperPixelContainer &container){
    cv::Mat greyImg = drwnGreyImage(img);
    // compute and quantize gradients
    pair<cv::Mat, cv::Mat> magAndOri = gradientMagnitudeAndOrientation(greyImg);
    const cv::Mat gradMagnitude(magAndOri.first);
    const cv::Mat gradOrientation(magAndOri.second);
    DRWN_ASSERT((gradMagnitude.data != NULL) && (gradOrientation.data != NULL));
    DRWN_ASSERT((gradMagnitude.rows == gradOrientation.rows) &&
        (gradMagnitude.cols == gradOrientation.cols));
    const int width = gradMagnitude.cols;
    const int height = gradMagnitude.rows;
    const int numSuperPixels = container.size();
    vector<cv::Mat> superPixelHistograms(_numOrientations);
    for (unsigned cnt = 0; cnt<_numOrientations; cnt++){
        superPixelHistograms[cnt] =  cv::Mat::zeros(numSuperPixels, 1, CV_32FC1);
    }
    vector<cv::Mat> maps;
    container.getMap(maps);
    // vote for super pixel
    for (int y = 0; y < height; y++) {
        const float *pm = gradMagnitude.ptr<const float>(y);
        const int *po = gradOrientation.ptr<int>(y);
        const int *ps = maps[0].ptr<int>(y);
        for (int x = 0; x < width; x++) {
            if(ps[x]<0) continue;
            float *pc = superPixelHistograms[po[x]].ptr<float>(ps[x]);
            pc[0] += pm[x];
        }
    }
    normalizeFeatureVectors(superPixelHistograms);
    if ((_clipping.first > 0.0) || (_clipping.second < 1.0)) {
        clipFeatureVectors(superPixelHistograms);
        normalizeFeatureVectors(superPixelHistograms);
    }
    features.resize(numSuperPixels);
    for(unsigned cnt = 0;cnt<_numOrientations; cnt++){
        for(unsigned row = 0; row<numSuperPixels;row++){
            const double* p = superPixelHistograms[cnt].ptr<double>(row);
            for(unsigned col = 0; col<1; col++){
                features[row*(1)+col].push_back(p[col]);
            }
        }
    }
}
