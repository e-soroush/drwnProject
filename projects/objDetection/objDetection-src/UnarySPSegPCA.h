#ifndef UNARYSPSEGPCA_H
#define UNARYSPSEGPCA_H
#include "UnarySegmentation.h"

class UnarySPSegPCA : public UnarySegmentation
{
public:
    UnarySPSegPCA();
    UnarySPSegPCA(const string &confAddress);
    ~UnarySPSegPCA();
    void Process();
    void makeTrainDataset();
    void makeTestDataset();
    void trainModel();
    void testModel();



private:
    void makeDataset(const string &imageListName, bool trainDataset);
};

#endif // UNARYSPSEGPCA_H
