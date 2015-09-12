#ifndef UNARYSEGMENTATION_H
#define UNARYSEGMENTATION_H
#include "drwnBase.h"
#include "drwnVision.h"
#include "SuperPixelContainer.h"
#include "HOGFeatures.h"

class UnarySegmentation
{
public:
    UnarySegmentation();
    ~UnarySegmentation();
    void Process();
    void initConfigXml();
    void readConfig();
    void makeTrainDataset();
    void makeTestDataset();
    void trainModel();
    void testModel();



protected:
    string _imgDir;
    string _lblDir;
    string _outputDir;
    string _baseDir;
    string _imgExt;
    string _lblExt;


    int _noClass;
    unsigned _gridSizeSP;
    int _usePreSuperPixels;

    string _trainList;
    string _testList;
    string _trainValList;
    string _superPixelAddress;

    string _trainDataset;
    string _testDataset;
    string _baseWorkDir;
    string _modelName;

    string _featureName;

    string _configAddress;

    drwnClassifierDataset _dataset;

private:
    void loadSuperpixelLabels(MatrixXi &labels,const char *lblFilename, SuperPixelContainer &container, vector<int> &targets);
    void makeDataset(const string &imageListName, bool trainDataset);

};

#endif // UNARYSEGMENTATION_H
