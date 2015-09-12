#include "UnarySegmentation.h"

UnarySegmentation::UnarySegmentation()
{
    _configAddress = "../objDetection-src/MSRC_example/MSRCConf.xml";

}

UnarySegmentation::~UnarySegmentation()
{

}

void UnarySegmentation::Process(){
    if(!drwnFileExists(_configAddress.c_str()))
        initConfigXml();
    readConfig();
    if(!drwnFileExists((_baseWorkDir+"unary"+_featureName+"TrainDataset.bin").c_str()))
        makeTrainDataset();
    if(!drwnFileExists((_baseWorkDir+"unary"+_featureName+"TestDataset.bin").c_str()))
        makeTestDataset();
    if(!drwnFileExists((_baseWorkDir+"unary"+_featureName+"TrainedModel.bin").c_str()))
        trainModel();
    testModel();
}

void UnarySegmentation::initConfigXml(){
    drwnXMLDoc xml;
    drwnXMLNode *rootNode, *addressInfo, *node, *regionInfo, *trainImages, *testImages;
    rootNode = drwnAddXMLRootNode(xml, "MSRC_Config", true);
    addressInfo = drwnAddXMLChildNode(*rootNode, "Addressing_informations");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node, "name", "baseDir");
    drwnAddXMLAttribute(*node, "value", "/home/ebi/Datasets/MSRC/");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "imageDir");
    drwnAddXMLAttribute(*node, "value", "Images/");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "labelDir");
    drwnAddXMLAttribute(*node, "value", "labels/");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "groundTruthDir");
    drwnAddXMLAttribute(*node, "value", "GroundTruth/");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "imgExt");
    drwnAddXMLAttribute(*node, "value", ".bmp");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "lblExt");
    drwnAddXMLAttribute(*node, "value", ".txt");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "groundTruthExt");
    drwnAddXMLAttribute(*node, "value", "_GT.bmp");
    node = drwnAddXMLChildNode(*addressInfo, "option");
    drwnAddXMLAttribute(*node,"name", "outputDir");
    drwnAddXMLAttribute(*node, "value", "output/");
    regionInfo = drwnAddXMLChildNode(*rootNode, "Region_Informarion");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "-1");
    drwnAddXMLAttribute(*node, "name", "void");
    drwnAddXMLAttribute(*node, "color", "0 0 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "0");
    drwnAddXMLAttribute(*node, "name", "building");
    drwnAddXMLAttribute(*node, "color", "128 0 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "1");
    drwnAddXMLAttribute(*node, "name", "grass");
    drwnAddXMLAttribute(*node, "color", "0 128 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "2");
    drwnAddXMLAttribute(*node, "name", "tree");
    drwnAddXMLAttribute(*node, "color", "128 128 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "3");
    drwnAddXMLAttribute(*node, "name", "cow");
    drwnAddXMLAttribute(*node, "color", "0 0 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "4");
    drwnAddXMLAttribute(*node, "name", "sheep");
    drwnAddXMLAttribute(*node, "color", "0 128 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "5");
    drwnAddXMLAttribute(*node, "name", "sky");
    drwnAddXMLAttribute(*node, "color", "128 128 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "6");
    drwnAddXMLAttribute(*node, "name", "airplane");
    drwnAddXMLAttribute(*node, "color", "192 0 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "7");
    drwnAddXMLAttribute(*node, "name", "water");
    drwnAddXMLAttribute(*node, "color", "64 128 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "8");
    drwnAddXMLAttribute(*node, "name", "face");
    drwnAddXMLAttribute(*node, "color", "192 128 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "9");
    drwnAddXMLAttribute(*node, "name", "car");
    drwnAddXMLAttribute(*node, "color", "64 0 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "10");
    drwnAddXMLAttribute(*node, "name", "bicycle");
    drwnAddXMLAttribute(*node, "color", "192 0 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "11");
    drwnAddXMLAttribute(*node, "name", "flower");
    drwnAddXMLAttribute(*node, "color", "64 128 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "12");
    drwnAddXMLAttribute(*node, "name", "sign");
    drwnAddXMLAttribute(*node, "color", "192 128 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "13");
    drwnAddXMLAttribute(*node, "name", "bird");
    drwnAddXMLAttribute(*node, "color", "0 64 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "14");
    drwnAddXMLAttribute(*node, "name", "book");
    drwnAddXMLAttribute(*node, "color", "128 64 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "15");
    drwnAddXMLAttribute(*node, "name", "chair");
    drwnAddXMLAttribute(*node, "color", "0 192 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "16");
    drwnAddXMLAttribute(*node, "name", "road");
    drwnAddXMLAttribute(*node, "color", "128 64 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "17");
    drwnAddXMLAttribute(*node, "name", "cat");
    drwnAddXMLAttribute(*node, "color", "0 192 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "18");
    drwnAddXMLAttribute(*node, "name", "dog");
    drwnAddXMLAttribute(*node, "color", "128 192 128");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "19");
    drwnAddXMLAttribute(*node, "name", "body");
    drwnAddXMLAttribute(*node, "color", "64 64 0");
    node = drwnAddXMLChildNode(*regionInfo, "region");
    drwnAddXMLAttribute(*node, "id", "20");
    drwnAddXMLAttribute(*node, "name", "boat");
    drwnAddXMLAttribute(*node, "color", "192 64 0");
    node=drwnAddXMLChildNode(*rootNode, "numClass", "21");
    trainImages=drwnAddXMLChildNode(*rootNode, "Train_Information");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "baseDir");
    drwnAddXMLAttribute(*node, "value", "/home/ebi/Projects/darwin/projects/objDetection/objDetection-src/MSRC_example/");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "trainList");
    drwnAddXMLAttribute(*node, "value", "trainList.txt");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "trainValList");
    drwnAddXMLAttribute(*node, "value", "trainValList.txt");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "testList");
    drwnAddXMLAttribute(*node, "value", "testList.txt");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "superPixelAddress");
    drwnAddXMLAttribute(*node, "value", "/home/ebi/Datasets/MSRC/regions/");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "usePreSuperPixels");
    drwnAddXMLAttribute(*node, "value", "0");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "featureName");
    drwnAddXMLAttribute(*node, "value", "HOG");
    node = drwnAddXMLChildNode(*trainImages, "option");
    drwnAddXMLAttribute(*node, "name", "gridSizeSuperPixel");
    drwnAddXMLAttribute(*node, "value", "15");


    ofstream fs(_configAddress);
    fs<<xml;
    fs.close();
    node->remove_all_nodes();
    regionInfo->remove_all_nodes();
    addressInfo->remove_all_nodes();
    rootNode->remove_all_nodes();
}

void UnarySegmentation::readConfig(){
    drwnXMLDoc confXml;
    drwnXMLNode *node = drwnParseXMLFile(confXml, _configAddress.c_str());
    drwnXMLNode *addressNode = node->first_node("Addressing_informations");
    for (drwnXMLNode *it = addressNode->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("imageDir")==0)
            _imgDir.assign(value->value());
        else if(string(name->value()).compare("baseDir")==0)
            _baseDir.assign(value->value());
        else if(string(name->value()).compare("labelDir")==0)
            _lblDir.assign(value->value());
        else if(string(name->value()).compare("outputDir")==0)
            _outputDir.assign(value->value());
        else if(string(name->value()).compare("imgExt")==0)
            _imgExt.assign(value->value());
        else if(string(name->value()).compare("lblExt")==0)
            _lblExt.assign(value->value());
    }
    drwnXMLNode *numClassNode = node->first_node("numClass");
    _noClass = atoi(numClassNode->value());
    drwnXMLNode *trainInfo = node->first_node("Train_Information");
    for (drwnXMLNode *it = trainInfo->first_node("option"); it != NULL; it = it->next_sibling("option")) {
        drwnXMLAttr *name = it->first_attribute("name");
        drwnXMLAttr *value = it->first_attribute("value");
        DRWN_ASSERT((name != NULL) && (value != NULL));
        if(string(name->value()).compare("baseDir")==0)
            _baseWorkDir.assign(value->value());
        else if(string(name->value()).compare("trainList")==0)
            _trainList.assign(value->value());
        else if(string(name->value()).compare("trainValList")==0)
            _trainValList.assign(value->value());
        else if(string(name->value()).compare("testList")==0)
            _testList.assign(value->value());
        else if(string(name->value()).compare("featureName")==0)
            _featureName.assign(value->value());
        else if(string(name->value()).compare("gridSizeSuperPixel")==0)
            _gridSizeSP = atoi(value->value());
        else if(string(name->value()).compare("superPixelAddress")==0)
            _superPixelAddress.assign(value->value());
        else if(string(name->value()).compare("usePreSuperPixels")==0)
            _usePreSuperPixels = atoi(value->value());
    }
}

void UnarySegmentation::makeTrainDataset(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Prepairing for making training dataset");
    const string imageListName = string(_baseWorkDir+_trainList);
    makeDataset(imageListName, true);
    _dataset.write((_baseWorkDir+"unary"+_featureName+"TrainDataset.bin").c_str());
}

void UnarySegmentation::makeTestDataset(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Prepairing for making test dataset");
    const string imageListName = string(_baseWorkDir+_testList);
    makeDataset(imageListName, false);
    _dataset.write((_baseWorkDir+"unary"+_featureName+"TestDataset.bin").c_str());
}

void UnarySegmentation::trainModel(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Training classifier");
    _dataset.read((_baseWorkDir+"unary"+_featureName+"TrainDataset.bin").c_str());
    int nFeatures = _dataset.numFeatures();
    int nClasses = _dataset.maxTarget() + 1;
    drwnBoostedClassifier model(nFeatures, nClasses);
    model.train(_dataset);
    model.write((_baseWorkDir+"unary"+_featureName+"TrainedModel.bin").c_str());
}

void UnarySegmentation::testModel(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("test the trained classifier");
    _dataset.read((_baseWorkDir+"unary"+_featureName+"TestDataset.bin").c_str());
    drwnBoostedClassifier model;
    model.read((_baseWorkDir+"unary"+_featureName+"TrainedModel.bin").c_str());
    vector<int> predictions;
    model.getClassifications(_dataset.features, predictions);
    double acc = 0;
    double totalSP = 0;
    for(unsigned cnt = 0; cnt<predictions.size(); cnt++){
        if(_dataset.targets[cnt]<0) continue;
        if(predictions[cnt]==_dataset.targets[cnt]) acc+=_dataset.weights[cnt];
        totalSP+=_dataset.weights[cnt];
    }

    cout<<"accuracy is: "<<acc/totalSP<<endl;
}

void UnarySegmentation::loadSuperpixelLabels(MatrixXi &labels, const char *lblFilename, SuperPixelContainer &container, vector<int> &targets){
    drwnLoadPixelLabels(labels, lblFilename, _noClass);
    vector<vector<int> > segIds;
    container.gTruthSuperPixel(segIds, labels);
    targets = segIds[0];
}

void UnarySegmentation::makeDataset(const string &imageListName, bool trainDataset){
    // retrieve image list base name without extension
    const char * imageList = imageListName.c_str();
    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }
    // read all images
    float processed = 0;
    for (unsigned cnt = 0; cnt<baseNames.size(); cnt++){
        string lblFilename = _baseDir + _lblDir + baseNames[cnt] + _lblExt;
        string imgFilename = _baseDir + _imgDir + baseNames[cnt] + _imgExt;
        cv::Mat img = imread(imgFilename);
        // container for super pixels
        SuperPixelContainer container;
        if (_usePreSuperPixels==0)
            container.addSuperpixels(drwnFastSuperpixels(img, _gridSizeSP));
        else
            container.loadSuperpixels((_superPixelAddress+baseNames[cnt]+".bin").c_str());
        vector<vector<double> > features;
        if(_featureName=="HOG"){
            HOGFeatures hogFeat;
            hogFeat.computeFeatures(img, features, container);
        }
        int nps = container.size();
        // always last super pixel map is groun truth
        container.loadSuperpixels(lblFilename.c_str());
        // retrieve label of each super pixel
        MatrixXi labels(img.rows, img.cols);
        vector<int> targets, rTarget;
        loadSuperpixelLabels(labels, lblFilename.c_str(), container, targets);
        vector<vector<double> > featureVector;
        // don't train unknown class -1
        if(trainDataset)
            for (unsigned cnt1 = 0; cnt1<features.size(); cnt1++){
                if(targets[cnt1]<0) continue;
                featureVector.push_back(features[cnt1]);
                rTarget.push_back(targets[cnt1]);
            }
        else{
            featureVector=features;
            rTarget = targets;
        }
        for (unsigned cnt1 = 0; cnt1<rTarget.size(); cnt1++)
            _dataset.append(featureVector[cnt1],rTarget[cnt1],container.pixels(cnt1));
        if((cnt % (int)(baseNames.size()/10))==0 ){
            DRWN_LOG_MESSAGE(processed*10<<" percent completed ...");
            processed++;
        }
    }
}
