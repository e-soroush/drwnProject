#include "drwnVision.h"
#include "drwnBase.h"
#include "opencv2/opencv.hpp"
#include "SuperPixelContainer.h"
#include "FeaturesContainer.h"
#include "HOGFeatures.h"
#include "UnarySegmentation.h"
#include "UnarySPSegPCA.h"

//string datasets_path = std::getenv("DATASETS_DIR");
    string datasets_path = "/home/soroush/Datasets";
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



void makeDataset(){
    const string filename = "../objDetection-src/MSRC_example/MSRCConf.xml";
    drwnXMLDoc confXml;
    drwnXMLNode *node = drwnParseXMLFile(confXml, filename.c_str());
    drwnXMLNode *addressNode = node->first_node("Addressing_informations");
    string imageDir, labelDir, imgExt, lblExt, baseDir, outputDir;
    int numClass;
    for (drwnXMLNode *it = addressNode->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("imageDir")==0)
            imageDir.assign(value->value());
        else if(string(name->value()).compare("baseDir")==0)
            baseDir.assign(value->value());
        else if(string(name->value()).compare("labelDir")==0)
            labelDir.assign(value->value());
        else if(string(name->value()).compare("outputDir")==0)
            outputDir.assign(value->value());
        else if(string(name->value()).compare("imgExt")==0)
            imgExt.assign(value->value());
        else if(string(name->value()).compare("lblExt")==0)
            lblExt.assign(value->value());
    }
    drwnXMLNode *numClassNode = node->first_node("numClass");
    numClass = atoi(numClassNode->value());
    drwnXMLNode *trainInfo = node->first_node("Train_Information");
    string trainList, testList, trainValList, baseDirT;
    for (drwnXMLNode *it = trainInfo->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("baseDir")==0)
            baseDirT.assign(value->value());
        else if(string(name->value()).compare("trainList")==0)
            trainList.assign(value->value());
        else if(string(name->value()).compare("trainValList")==0)
            trainValList.assign(value->value());
        else if(string(name->value()).compare("testList")==0)
            testList.assign(value->value());
    }
    const string tmp = string(baseDirT+trainList);
    const char * imageList = tmp.c_str();
    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }
    DRWN_LOG_MESSAGE("Prepairing Train data ...");
    drwnClassifierDataset dataset;
    float processed = 0;
    for (unsigned cnt = 0; cnt<baseNames.size(); cnt++){
        string lblFilename = (baseDir + labelDir + baseNames[cnt]+lblExt);
        string imgFilename = baseDir + imageDir + baseNames[cnt] + imgExt;
        cv::Mat img = imread(imgFilename);
        SuperPixelContainer container;
        container.addSuperpixels(drwnFastSuperpixels(img, 15));
        vector<int> rTarget;
        HOGFeatures hogFeat;
        vector<vector<double> > features;
        hogFeat.computeFeatures(img, features, container);
//        int noSps = container.size();
        container.loadSuperpixels(lblFilename.c_str());
        MatrixXi labels(img.rows, img.cols);
        drwnLoadPixelLabels(labels, lblFilename.c_str(), numClass);
        vector<vector<int> > segIds;
        container.gTruthSuperPixel(segIds, labels);
        vector<int> targets = segIds[0];
        vector<vector<double> > featureVector;
        for (unsigned cnt1 = 0; cnt1<features.size(); cnt1++){
            if(targets[cnt1]<0) continue;
            if(targets[cnt1]>20)
                cout<<"why??"<<endl;
            featureVector.push_back(features[cnt1]);
            rTarget.push_back(targets[cnt1]);
        }
        for (unsigned cnt1 = 0; cnt1<rTarget.size(); cnt1++)
            dataset.append(featureVector[cnt1],rTarget[cnt1]);
        if((cnt % (int)(baseNames.size()/10))==0 ){
            DRWN_LOG_MESSAGE(processed*10<<" percent completed ...");
            processed++;
        }

    }
    dataset.write((baseDirT+string("SPHOGFeatureDataset.bin")).c_str());
    int nFeatures = dataset.numFeatures();
    int nClasses = dataset.maxTarget() + 1;
    drwnBoostedClassifier model(nFeatures, nClasses);
    model.train(dataset);
    model.write((baseDirT+string("SPHOGFeatureModel.xml")).c_str());
}
void trainModel(){
    drwnClassifierDataset dataset;
    string base_dir = "/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/";
    string dataset_dir = base_dir+"SPHOGFeatureDataset.bin";
    string model_dir = base_dir + "SPHOGFeatureModel.xml";
    dataset.read(dataset_dir.c_str());
    int nFeatures = dataset.numFeatures();
    int nClasses = dataset.maxTarget() + 1;
    drwnBoostedClassifier model(nFeatures, nClasses);
    model.train(dataset);
    model.write((model_dir).c_str());
}
void testModel(){
    const string filename = "../objDetection-src/MSRC_example/MSRCConf.xml";
    drwnXMLDoc confXml;
    drwnXMLNode *node = drwnParseXMLFile(confXml, filename.c_str());
    drwnXMLNode *addressNode = node->first_node("Addressing_informations");
    string imageDir, labelDir, imgExt, lblExt, baseDir, outputDir;
    int numClass;
    for (drwnXMLNode *it = addressNode->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("imageDir")==0)
            imageDir.assign(value->value());
        else if(string(name->value()).compare("baseDir")==0)
            baseDir.assign(value->value());
        else if(string(name->value()).compare("labelDir")==0)
            labelDir.assign(value->value());
        else if(string(name->value()).compare("outputDir")==0)
            outputDir.assign(value->value());
        else if(string(name->value()).compare("imgExt")==0)
            imgExt.assign(value->value());
        else if(string(name->value()).compare("lblExt")==0)
            lblExt.assign(value->value());
    }
    drwnXMLNode *numClassNode = node->first_node("numClass");
    numClass = atoi(numClassNode->value());
    drwnXMLNode *trainInfo = node->first_node("Train_Information");
    string trainList, testList, trainValList, baseDirT;
    for (drwnXMLNode *it = trainInfo->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("baseDir")==0)
            baseDirT.assign(value->value());
        else if(string(name->value()).compare("trainList")==0)
            trainList.assign(value->value());
        else if(string(name->value()).compare("trainValList")==0)
            trainValList.assign(value->value());
        else if(string(name->value()).compare("testList")==0)
            testList.assign(value->value());
    }
    const string tmp = string(baseDirT+testList);
    const char * imageList = tmp.c_str();
    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }
    DRWN_LOG_MESSAGE("Prepairing Train data ...");
    drwnClassifierDataset dataset;
    float processed = 0;
    for (unsigned cnt = 0; cnt<baseNames.size(); cnt++){
        string lblFilename = (baseDir + labelDir + baseNames[cnt]+lblExt);
        string imgFilename = baseDir + imageDir + baseNames[cnt] + imgExt;
        cv::Mat img = imread(imgFilename);
        SuperPixelContainer container;
        container.addSuperpixels(drwnFastSuperpixels(img, 15));
        vector<int> rTarget;
        HOGFeatures hogFeat;
        vector<vector<double> > features;
        hogFeat.computeFeatures(img, features, container);
//        int noSps = container.size();
        container.loadSuperpixels(lblFilename.c_str());
        MatrixXi labels(img.rows, img.cols);
        drwnLoadPixelLabels(labels, lblFilename.c_str(), numClass);
        vector<vector<int> > segIds;
        container.gTruthSuperPixel(segIds, labels);
        vector<int> targets = segIds[0];
        vector<vector<double> > featureVector;
        for (unsigned cnt1 = 0; cnt1<targets.size(); cnt1++)
            dataset.append(features[cnt1],targets[cnt1]);
        if((cnt % (int)(baseNames.size()/10))==0 ){
            DRWN_LOG_MESSAGE(processed*10<<" percent completed ...");
            processed++;
        }

    }
    dataset.write((baseDirT+string("SPHOGFeatureTest.bin")).c_str());
    int nFeatures = dataset.numFeatures();
    int nClasses = dataset.maxTarget() + 1;
    drwnBoostedClassifier model;
    string model_dir = (baseDirT)+string("SPHOGFeatureModel.xml");
    model.read(model_dir.c_str());
    vector<int> predictions;
    model.getClassifications(dataset.features, predictions);
    double acc = 0;
    for(unsigned cnt = 0; cnt<predictions.size(); cnt++){
        if(predictions[cnt]==dataset.targets[cnt]) acc++;
    }
    cout<<"accuracy is: "<<acc/predictions.size()<<endl;
}

void testEModel(){
    drwnClassifierDataset dataset;
    drwnBoostedClassifier model;
    string base_dir = "/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/";
    string dataset_dir = base_dir+"SPHOGFeatureTest.bin";
    string model_dir = base_dir + "SPHOGFeatureModel.xml";
    dataset.read(dataset_dir.c_str());
    model.read(model_dir.c_str());
    vector<int> predictions;
    model.getClassifications(dataset.features, predictions);
    double acc = 0;
    double total = 0;
    for(unsigned cnt = 0; cnt<predictions.size(); cnt++){
        if(dataset.targets[cnt]<0) continue;
        if(predictions[cnt]==dataset.targets[cnt]) acc++;
        total++;
    }
    cout<<"accuracy is: "<<acc/total<<endl;
}


int main(int argc, char * argv[]){
//    string datasets_path = std::getenv("DATASETS_DIR");
//    //    string datasets_path = "/home/ebi/Datasets";
//    const char *baseName = "1_19_s";
//    cv::Mat img = cv::imread(datasets_path+"/MSRC/Images/" + baseName + ".bmp");

//    drwnSegImageInstance instance(img, baseName);
//    instance.appendPixelFeatures();
//    vector<vector<double> > unaries =  instance.unaries;
    UnarySPSegPCA unarySegmentation("secConfig.xml");
    unarySegmentation.Process();
//    unarySegmentation.initConfigXml();
//    unarySegmentation.readConfig();
//    unarySegmentation.makeTrainDataset();
//    unarySegmentation.trainModel();
//    unarySegmentation.makeTestDataset();
//    unarySegmentation.testModel();

//    drwnClassifierDataset dataset;
//    dataset.read("/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/unaryPCATrainDataset.bin");
//    vector<double> features = dataset.features[0];
//    int target = dataset.targets[0];
//    vector<drwnBoostedClassifier> unaryModel;
//    unaryModel.resize(21);
//    string path = "/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/unaryPCATrainedModel";
//    for (unsigned cnt = 0; cnt<21; cnt++){
//        unaryModel[cnt].read((path+to_string(cnt)+".bin").c_str());
//    }
//    vector<int> classId;
//    vector<double> confidence;
//    for (unsigned cnt = 0; cnt< 21; cnt++){
//        drwnBoostedClassifier model = unaryModel[cnt];
//        classId.push_back(model.getClassification(features));
//        vector<double> confs;
//        model.getClassScores(features, confs);
//        confidence.push_back(confs[classId.back()]);
//    }
//    cout <<"classId: "<<endl;


    //    testHOG();
    //    testDataset();
    //    testClassifier();
    //    testSPWithClassifier();
    //    configDataset();
    //    makeDataset();
    //    trainModel();
    //    testModel();
    //    testEModel();
    //    UnarySegmentation unary;
    //    unary.Process();
    //    drwnFeatureWhitener pixelFeatureWhitener;
    //    drwnClassifierDataset dataset;
    //    string base_dir = "/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/";
    //    string dataset_dir = base_dir+"unaryHOGTrainDataset.bin";
    //    string model_dir = base_dir + "unaryHOGTrainedModel.bin";
    //    dataset.read(dataset_dir.c_str());
    //    int numClasses = dataset.maxTarget()+1;
    //    int numFeatures = dataset.numFeatures();
    //    pixelFeatureWhitener.train(dataset.features);
    //    pixelFeatureWhitener.transform(dataset.features);
    //    drwnBoostedClassifier model(numFeatures, numClasses);
    //    drwnConfusionMatrix confusion(numClasses);
    //    vector<int> predictions;
    //    model.train(dataset);
    //    model.getClassifications(dataset.features, predictions);;
    //    confusion.accumulate(dataset.targets, predictions);
    //    confusion.write(cout);
    //    confusion.avgPrecision();
    //    cout<<"avgPer: " << confusion.avgPrecision()<<endl;
    //    cout<<"avgRecal: " << confusion.avgRecall()<<endl;
    //    cout<<"Accuracy: " << confusion.accuracy()<<endl;
    //    model.read(model_dir.c_str());

    //    vector<cv::Mat> selected,response, textoneFeatures;
    //    vector<vector<double> >features;
    //    SuperPixelContainer container;
    //    SuperPixelContainer gtContainer;
    //    drwnLBPFilterBank lbpFiltBank(true);
    //    drwnTextonFilterBank textoneFilter;
    //    FeaturesContainer featureContainer;
    //    textoneFilter.filter(img,textoneFeatures);
    //    container.addSuperpixels(drwnFastSuperpixels(img, 15));
    //    vector<int> tSize;
    //    for(int cnt = 0; cnt<container.size(); cnt++){
    //        set<unsigned> nbrs = container.neighbours(cnt);
    //        tSize.push_back(nbrs.size());
    //    }
    //    int maxTSize = *min_element(tSize.begin(), tSize.end());
    //    HOGFeatures hogFeat;
    //    hogFeat.computeFeatures(img, features, container);
    //    gtContainer.loadSuperpixels("/home/ebi/Datasets/MSRC/labels/1_19_s.txt");
    //    cv::imshow("gt",gtContainer.visualize(img,false));
    //    waitKey();

    //    //    cv::imshow("test", container.selectSuperPixel(20,img, selected));
    //    //    container.setMap(selected);
    //    lbpFiltBank.filter(img, response);
    //    lbpFiltBank.regionFeatures(container, response, features);
    //    featureContainer.setFeatures(textoneFeatures, container);
    //    //    featureContainer.appendFeatures(textoneFeatures, selected[0]);
    //    cv::imshow("test2",container.visualize(img,false));
    //    Mat summ = (gtContainer.visualize(Mat(img.size(), img.type(),Scalar::all(0)),false,true) + container.visualize(img,false));
    //    imshow("sum", summ);
    //    cv::waitKey();


//    drwnSegImageInstance instance(img);
//    drwnSegImageStdPixelFeatures stdPixelFeatures;
//    stdPixelFeatures.cacheInstanceData(instance);
//    cout<< "numFeat:" << stdPixelFeatures.numFeatures()<<endl;
}
