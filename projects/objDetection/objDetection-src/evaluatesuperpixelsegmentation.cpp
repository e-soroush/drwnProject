#include "drwnBase.h"
#include "drwnVision.h"
#include "SuperPixelContainer.h"

int main(int argc, char* argv[]){
    int channel = 2;
    int noClass = 21;
    // base path
    const string pathBase = "/home/ebi/Datasets/MSRC/";
    // path for super pixels stored
    const string pathRegionBase = pathBase + "Regions/";
    // path for images stored
    const string pathImageBase  = pathBase + "Images/";
    // path fro ground truth base
    const string pathLableBase = pathBase + "Labels/";
    // image extensions : bmp ot jpg
    const string imgExt = ".bmp";
    // super pixel extension
    const string spExt  = ".spx";
    const string lblExt = ".txt";
    // ia text file contain image file names
    const string imageListName = pathBase + "imgLists.txt";
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
    int error = 0;
    int total = 0;
    int processed = 0;
    DRWN_LOG_MESSAGE("Processing images ...");
    for(string baseName : baseNames){
        // read superpixel files and labels
        SuperPixelContainer container;
        const string spName = pathRegionBase + baseName + spExt;
        const string lblFilename = pathLableBase +baseName + lblExt;
        MatrixXi labels;
        // assign labels to matrix on eigen
        drwnLoadPixelLabels(labels, lblFilename.c_str(), noClass);
        ifstream ifs(spName, ios::binary);
        container.read(ifs);
        ifs.close();
        // find label of each super pixel
        for(unsigned ID = 0; ID<container.cSize(channel); ID++){
            vector<Point> pixels;
            container.getPixelsByID(channel, ID, pixels);
            if(pixels.size()==0) continue;
            vector<Mat> maps;
            container.getMap(maps);
            vector<int> classIds;
            for(Point pixel:pixels){
                classIds.push_back(labels(pixel.y,pixel.x));
            }
            // highest vote for each superpixel
            sort(classIds.begin(), classIds.end());
            int maxId = classIds[classIds.size()/2];
            // evaluate with ground truth
            for(Point pixel:pixels){
                if(labels(pixel.y, pixel.x) == -1) continue;
                if(labels(pixel.y, pixel.x) != maxId) error++;
                total++;
            }
        }
        processed++;
        int base10 = baseNames.size()/10;
        if((processed % base10) == 0){
            DRWN_LOG_MESSAGE("Processed :"<<(float)processed/(float)baseName.size() << "%");
        }
    }
    DRWN_LOG_MESSAGE("error: "<<(float)error/(float)total);
}
