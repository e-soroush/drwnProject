#include "drwnBase.h"
#include "drwnVision.h"
#include "SuperPixelContainer.h"

// create super pixel and save it to a directory
int main(int argc, char * argv[]){
    // base path
    const string pathBase = "/home/soroush/Datasets/MSRC/";
    // path for super pixels stored
    const string pathRegionBase = pathBase + "Regions/";
    // path for images stored
    const string pathImageBase  = pathBase + "Images/";
    // image extensions : bmp ot jpg
    const string imgExt = ".bmp";
    // super pixel extension
    const string spExt  = ".spx";
    bool appendFile = false;
    // ia text file contain image file names
    const string imageListName = pathBase + "imgLists.txt";
    // grid sizes fot super pixel
    vector<unsigned> gridSizes={10, 15, 20};
    const char * imageList = imageListName.c_str();
    // read base image and super pixel file name
    vector<string> baseNames;
    if (drwnFileExists(imageList)) {
        DRWN_LOG_MESSAGE("Reading image list from " << imageList << "...");
        baseNames = drwnReadFile(imageList);
        DRWN_LOG_MESSAGE("...read " << baseNames.size() << " images");
    } else {
        DRWN_LOG_MESSAGE("Processing single image " << imageList << "...");
        baseNames.push_back(string(imageList));
    }
    DRWN_LOG_MESSAGE("Start super pixel extracting in  " << gridSizes.size() << " grid sizes" );
    int processed = 0;
    for(string baseName : baseNames){
        cv::Mat img = imread(pathImageBase+baseName+imgExt);
        SuperPixelContainer container;
        const string spName = pathRegionBase + baseName + spExt;
        // append super new super pixels to existing files
        if(drwnFileExists(spName.c_str()) && appendFile){
            ifstream ifs(spName, ios::binary);
            container.read(ifs);
            ifs.close();
        }
        for(auto gridSize:gridSizes)
            container.addSuperpixels(drwnFastSuperpixels(img, gridSize));
        // save to binary file
        ofstream ofs (spName, ios::binary);
        container.write(ofs);
        ofs.close();
        if((++processed)%((int)baseNames.size()/10)==0)
            DRWN_LOG_MESSAGE(processed/baseNames.size()*100<<" processed.");
    }
}
