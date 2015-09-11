#include "drwnVision.h"
#include "opencv2/opencv.hpp"
#include "SuperPixelContainer.h"
#include "FeaturesContainer.h"
#include "HOGFeatures.h"

string datasets_path = std::getenv("DATASETS_DIR");
//    string datasets_path = "/home/ebi/Datasets";
cv::Mat img = cv::imread(datasets_path+"/MSRC/Images/1_19_s.bmp");

void testHOG(){
    drwnHOGFeatures hogFeat;
    std::vector<cv::Mat> features, denseFeatures;
    int numFeature = hogFeat.numFeatures();

    hogFeat.computeFeatures(img, features);
    hogFeat.computeDenseFeatures(img, denseFeatures);
    cv::Mat cellVis = hogFeat.visualizeCells(img);
    vector<Mat> views;
    views.push_back(cellVis);
    views.push_back(img);
    imshow("cellVis", drwnCombineImages(views));
    imshow("HOGDense",drwnCombineImages(features));
    waitKey();
}

void testDataset(){
    drwnHOGFeatures hogFeat;
    std::vector<cv::Mat> features;
    hogFeat.computeFeatures(img, features);
    vector<vector<double> >featureVector;
    vector<int> targets;
    featureVector.resize(features[0].rows*features[0].cols);
    targets.resize(featureVector.size());
    for(unsigned cnt = 0;cnt<features.size(); cnt++){
        for(unsigned row = 0; row<features[cnt].rows;row++){
            const double* p = features[cnt].ptr<double>(row);
            for(unsigned col = 0; col<features[cnt].cols; col++){
                featureVector[row*(features[cnt].cols)+col].push_back(p[col]);
                targets[row*(features[cnt].cols)+col]=(row*(features[cnt].cols)+col)/(targets.size()/2);// 2 class test
            }
        }
    }
    drwnDataset<double,int,double> dataset;
    for (unsigned cnt = 0; cnt<targets.size(); cnt++)
        dataset.append(featureVector[cnt],targets[cnt]);
    dataset.write("testDataset.bin");
}

void testClassifier(){
    drwnClassifierDataset dataset;
    dataset.read("testDataset.bin");
    const int nFeatures = dataset.numFeatures();
    const int nClasses = dataset.maxTarget() + 1;
    drwnBoostedClassifier model(nFeatures, nClasses);
    model.train(dataset);
    vector<int> predictions;
    model.getClassifications(dataset.features, predictions);
    int i = 0 ;
}

void testSPWithClassifier(){
    SuperPixelContainer container;
    SuperPixelContainer gtContainer;
    container.addSuperpixels(drwnFastSuperpixels(img, 15));
    gtContainer.loadSuperpixels("/home/ebi/Datasets/MSRC/labels/1_19_s.txt");
    vector<Mat> maps;
    gtContainer.getMap(maps);
    container.addSuperpixels(maps[0]);
//    imshow("hh",container.visualize(img, false));
//    waitKey();
    string lblFilename = (string)getenv("DATASETS_DIR")+"/MSRC/labels/1_19_s.txt";
    MatrixXi labels(img.rows, img.cols);
    int nLabels = 21;
    drwnLoadPixelLabels(labels, lblFilename.c_str(), nLabels);
    vector<vector<int> > segIds;
    container.gTruthSuperPixel(segIds, labels);
    vector<int> targets = segIds[0];
    vector<int> rTarget;
    HOGFeatures hogFeat;
    vector<vector<double> > features;
    hogFeat.computeFeatures(img, features, container);
    vector<vector<double> > featureVector;
    for (unsigned cnt = 0; cnt<features.size(); cnt++){
        if(targets[cnt]==-1) continue;
        featureVector.push_back(features[cnt]);
        rTarget.push_back(targets[cnt]);
    }
    drwnClassifierDataset dataset;
    for (unsigned cnt = 0; cnt<rTarget.size(); cnt++)
        dataset.append(featureVector[cnt],rTarget[cnt]);
    dataset.write("SPHOGFeatureDataset.bin");
    int nFeatures = dataset.numFeatures();
    int nClasses = dataset.maxTarget() + 1;
    drwnBoostedClassifier model(nFeatures, nClasses);
    model.train(dataset);
    SuperPixelContainer container2;
    cv::Mat img2 = imread((string)datasets_path+"/MSRC/Images/1_20_s.bmp");
    container2.addSuperpixels(drwnFastSuperpixels(img2, 15));
    string lblFilename2 = (string)getenv("DATASETS_DIR")+"/MSRC/labels/1_20_s.txt";
    container2.loadSuperpixels(lblFilename2.c_str());
    vector<vector<double> > features2;
    hogFeat.computeFeatures(img2, features2, container2);
    MatrixXi labels2(img.rows, img.cols);
    drwnLoadPixelLabels(labels2, lblFilename2.c_str(), nLabels);
    vector<vector<int> > segIds2;
    container2.gTruthSuperPixel(segIds2, labels2);
    vector<int> targets2 = segIds2[0];
    dataset.clear();
    for (unsigned cnt = 0; cnt<targets2.size(); cnt++)
        dataset.append(features2[cnt],targets2[cnt]);
    dataset.write("SPHOGFeatureTest.bin");
    vector<int> predictions;
    model.getClassifications(dataset.features, predictions);
    float acc = 0;
    int ttoal=0;
    for(unsigned cnt = 0; cnt<predictions.size(); cnt++){
        if(dataset.targets[cnt]==-1) continue;
        if(predictions[cnt]==dataset.targets[cnt]) {
            acc++;
        }
        ttoal++;
    }
    cout << "acuracy is: "<<acc/ttoal<<endl;
}


int main(int argc, char * argv[]){
//    testHOG();
//    testDataset();
//    testClassifier();
    testSPWithClassifier();
    vector<cv::Mat> selected,response, textoneFeatures;
    vector<vector<double> >features;
    SuperPixelContainer container;
    SuperPixelContainer gtContainer;
    drwnLBPFilterBank lbpFiltBank(true);
    drwnTextonFilterBank textoneFilter;
    FeaturesContainer featureContainer;
    textoneFilter.filter(img,textoneFeatures);
    container.addSuperpixels(drwnFastSuperpixels(img, 15));
    HOGFeatures hogFeat;
    hogFeat.computeFeatures(img, features, container);
    gtContainer.loadSuperpixels("/home/ebi/Datasets/MSRC/labels/1_19_s.txt");
    cv::imshow("gt",gtContainer.visualize(img,false));
    waitKey();
//    cv::imshow("test", container.selectSuperPixel(20,img, selected));
//    container.setMap(selected);
    lbpFiltBank.filter(img, response);
    lbpFiltBank.regionFeatures(container, response, features);
    featureContainer.setFeatures(textoneFeatures, container);
//    featureContainer.appendFeatures(textoneFeatures, selected[0]);
    cv::imshow("test2",container.visualize(img,false));
    Mat summ = (gtContainer.visualize(Mat(img.size(), img.type(),Scalar::all(0)),false,true) + container.visualize(img,false));
    imshow("sum", summ);
    cv::waitKey();
}
