#include "UnarySPSegPCA.h"

UnarySPSegPCA::UnarySPSegPCA()
{

}

UnarySPSegPCA::UnarySPSegPCA(const string &confAddress){
    _configAddress = confAddress;
}

UnarySPSegPCA::~UnarySPSegPCA()
{

}

void UnarySPSegPCA::Process(){
    if(!drwnFileExists(_configAddress.c_str()))
        initConfigXml();
    readConfig();
    if(!drwnFileExists((_baseWorkDir+"unary"+_methodName+"TrainDataset.bin").c_str()))
        makeTrainDataset();
    if(!drwnFileExists((_baseWorkDir+"unary"+_methodName+"TestDataset.bin").c_str()))
        makeTestDataset();
    if(!drwnFileExists((_baseWorkDir+"unary"+_methodName+"TrainedModel0.bin").c_str()))
        trainModel();
    testModel();
}

void UnarySPSegPCA::makeTrainDataset(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Prepairing for making training dataset");
    const string imageListName = string(_baseWorkDir+_trainList);
    makeDataset(imageListName, true);
    _dataset.write((_baseWorkDir+"unary"+_methodName+"TrainDataset.bin").c_str());
}

void UnarySPSegPCA::makeTestDataset(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Prepairing for making test dataset");
    const string imageListName = string(_baseWorkDir+_testList);
    makeDataset(imageListName, true);
    _dataset.write((_baseWorkDir+"unary"+_methodName+"TestDataset.bin").c_str());
}

void UnarySPSegPCA::trainModel(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("Training classifier");
    _dataset.read((_baseWorkDir+"unary"+_methodName+"TrainDataset.bin").c_str());
    int nFeatures = _dataset.numFeatures();
    int nClasses = _dataset.maxTarget() + 1;
    vector<drwnBoostedClassifier*> unaryModel(nClasses);
    for(unsigned cnt = 0; cnt<nClasses; cnt++){
        DRWN_LOG_MESSAGE("Training classifier for class" << cnt);
        drwnClassifierDataset tmpDataset;
        for(unsigned ins = 0; ins<_dataset.targets.size(); ins++){
            int target = _dataset.targets[ins] == cnt ? 1 : 0;
            tmpDataset.append(_dataset.features[ins], target);
        }
        unaryModel[cnt] = new drwnBoostedClassifier(nFeatures, 2);
        unaryModel[cnt]->train(tmpDataset);
        unaryModel[cnt]->write((_baseWorkDir+"unary"+_methodName+"TrainedModel" + to_string(cnt) + ".bin").c_str());
    }
}


void UnarySPSegPCA::testModel(){
    _dataset.clear();
    DRWN_LOG_MESSAGE("test the trained classifier");
    _dataset.read((_baseWorkDir+"unary"+_methodName+"TestDataset.bin").c_str());
    vector<drwnBoostedClassifier* > unaryModel;
    unaryModel.resize(_dataset.maxTarget()+1);
    const string path = _baseWorkDir+"unary"+_methodName+"TrainedModel";
    //read trained classifier model
    for (unsigned cnt = 0; cnt<unaryModel.size(); cnt++){
        unaryModel[cnt] = new drwnBoostedClassifier();
        unaryModel[cnt]->read((path+to_string(cnt)+".bin").c_str());
    }
    int total = 0;
    int acc = 0;
    for(unsigned cnt = 0; cnt<_dataset.targets.size(); cnt++){
        vector<int> classId;
        vector<double> confidence;
        for(unsigned um = 0; um < unaryModel.size(); um++){
            drwnBoostedClassifier *model = unaryModel[um];
            classId.push_back(model->getClassification(_dataset.features[cnt]));
            vector<double> confs;
            model->getClassScores(_dataset.features[cnt], confs);
            confidence.push_back(confs[classId.back()]);
        }
        double maxConf = 0;
        int classID = -1;
        for(unsigned sel = 0; sel<classId.size(); sel++){
            if(classId[sel]==1 && confidence[sel]>maxConf){
                classID = sel;
                maxConf = confidence[sel];
            }
        }
        if(classID == -1){
            double minConf = 100;
            for(unsigned i = 0; i<confidence.size(); i++){
                if (confidence[i]<minConf){
                    classID = i;
                    minConf = confidence[i];
                }

            }
        }
        if (_dataset.targets[cnt]==-1){continue;}
        if (_dataset.targets[cnt]==classID){acc++;}
        total++;
    }
    DRWN_LOG_MESSAGE("Total Accuracy in Superpixel is: "<<(double)acc/(double)total);

}


void UnarySPSegPCA::makeDataset(const string &imageListName, bool trainDataset){
    // retrieve image list base name without extension
    const char * imageList = imageListName.c_str();
    vector<string> baseNames;
    int channel = 1;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }
    // read all images
    int processed = 0;
    for (unsigned cnt = 0; cnt<baseNames.size(); cnt++){
        const string lblFilename = _baseDir + _lblDir + baseNames[cnt] + _lblExt;
        const string imgFilename = _baseDir + _imgDir + baseNames[cnt] + _imgExt;
        cv::Mat img = imread(imgFilename);
        // container for super pixels
        SuperPixelContainer container;
        if (_usePreSuperPixels==0)
            container.addSuperpixels(drwnFastSuperpixels(img, _gridSizeSP));
        else {
            const string tName = _superPixelAddress+baseNames[cnt]+_spExt;
            ifstream ifs(tName, ios::binary);
            container.read(ifs);
            ifs.close();
        }
        drwnSegImageInstance instance(img, imgFilename.c_str());
        instance.appendPixelFeatures();
        MatrixXi labels(img.rows, img.cols);
        vector<int> targets;
        loadSuperpixelLabels(labels, lblFilename.c_str(), container, targets);
        for(unsigned ID = 0; ID<container.cSize(channel); ID++){
            vector<double> features(instance.unaries[0].size(), 0);
            vector<Point> pixels;
            container.getPixelsByID(channel, ID, pixels);
            if(pixels.size()==0) continue;
            vector<int> classIds;
            for(Point pixel:pixels){
                // retrieve each pixel class
                classIds.push_back(labels(pixel.y,pixel.x));
                // retrieve eache pixel feature
                vector<double> tmp = instance.unaries[instance.pixel2Indx(pixel)];
                for(unsigned cnt = 0; cnt<tmp.size(); cnt++)
                    features[cnt] += tmp[cnt];
            }
            // highest vote for each superpixel
            sort(classIds.begin(), classIds.end());
            int maxId = classIds[classIds.size()/2];
            if(trainDataset){
                if (maxId == -1)
                    continue;
            }
            _dataset.append(features, maxId);
        }
        int base10 = baseNames.size()/10;
        processed++;
        if((processed%base10)==0)
            DRWN_LOG_MESSAGE("Processed: "<<(float)processed/(float)baseNames.size() * 100<<" %");
    }
}

