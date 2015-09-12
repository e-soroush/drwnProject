#include "SuperPixelContainer.h"

SuperPixelContainer::SuperPixelContainer()
{
}

SuperPixelContainer::~SuperPixelContainer()
{

}

cv::Mat SuperPixelContainer::selectSuperPixel(unsigned segId, const Mat& img, vector<cv::Mat> &selected){
    selected.insert(selected.begin(),_maps.begin(),_maps.end());
    for (unsigned i = 0; i < _maps.size(); i++) {
        for (int y = 0; y < _maps[i].rows; y++) {
            int *p = selected[i].ptr<int>(y);
            for (int x = 0; x < _maps[i].cols; x++) {
                if (p[x] != (int)segId) p[x] = -1;
            }
        }
    }
    // show selected superpixel
    float alpha=0.9;
    vector<cv::Mat> views;
    for (unsigned i = 0; i < selected.size(); i++) {
        views.push_back(img.clone());
        // colour pixels
        for (int y = 0; y < selected[i].rows; y++) {
            const int *p = selected[i].ptr<const int>(y);
            unsigned char * const q = views.back().ptr<unsigned char>(y);
            for (int x = 0; x < selected[i].cols; x++) {
                const cv::Scalar c = p[x] > 0 ? Scalar(1,1,1) : cv::Scalar::all(0);
                q[3 * x + 2] = (unsigned char)((alpha) * q[3 * x + 2] * c.val[2]);
                q[3 * x + 1] = (unsigned char)((alpha) * q[3 * x + 1] * c.val[1]);
                q[3 * x + 0] = (unsigned char)((alpha) * q[3 * x + 0] * c.val[0]);
            }
        }
    }

    return drwnCombineImages(views);
}

int SuperPixelContainer::getSuperPixelId(int x, int y){
    return _maps[0].at<int>(y,x);
}

// visualization
cv::Mat SuperPixelContainer::visualize(const cv::Mat& img, bool bColorById, bool gTruth) const
{
    DRWN_ASSERT(img.data != NULL);
    if (empty()) return img.clone();

    vector<cv::Mat> views;
    cv::Mat m(height(), width(), CV_8UC1);
    for (unsigned i = 0; i < _maps.size(); i++) {
        if (bColorById) {
            views.push_back(drwnCreateHeatMap(_maps[i]));
        } else {
            views.push_back(img.clone());
        }
        cv::compare(_maps[i], cv::Scalar::all(0.0), m, CV_CMP_LT);
        views.back().setTo(cv::Scalar::all(0.0), m);
        drwnShadeRegion(views.back(), m, CV_RGB(255, 0, 0), 1.0, DRWN_FILL_DIAG, 1);
        drwnDrawRegionBoundaries(views.back(), _maps[i], CV_RGB(255, 255, 255), 3);
        if(gTruth)
            drwnDrawRegionBoundaries(views.back(), _maps[i], CV_RGB(0, 255, 0), 1);
        else
            drwnDrawRegionBoundaries(views.back(), _maps[i], CV_RGB(255, 0, 0), 1);
    }

    return drwnCombineImages(views);
}

void SuperPixelContainer::gTruthSuperPixel(vector<vector<int> > &segIds, const MatrixXi &labels){

    cv::Mat map = _maps.back(); // ground truth
    double numSegs;double tmp1;Point p;
    minMaxLoc( map, &tmp1, &numSegs, &p, &p);
    segIds.clear();
    if (_maps.size()<2) return;
    segIds.resize(_maps.size()-1);
    for(unsigned cnt = 0; cnt<_maps.size()-1; cnt++){
        double numSegs;double tmp1;Point p;
        minMaxLoc( _maps[cnt], &tmp1, &numSegs, &p, &p);
        segIds[cnt].resize(numSegs+1);
        for(unsigned seg = 0; seg<numSegs+1; seg++){
            vector<int> tmp;
            for(unsigned row = 0; row<map.rows; row++){
                const int *ps = _maps[cnt].ptr<int>(row);
                for(unsigned col = 0; col<map.cols; col++)
                    if(ps[col]==seg)
                        tmp.push_back(labels(row, col));
            }
            if(tmp.empty())
                continue;
            std::sort(tmp.begin(), tmp.end());
            segIds[cnt][seg]=tmp[tmp.size()/2];
        }
    }
}
