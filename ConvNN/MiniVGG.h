// MiniVGG.h
#pragma once

#ifndef MINI_VGG_CONVNN_H
#define MINI_VGG_CONVNN_H

#include "include.h"
#include "Layer.h"

typedef struct cl_ConvLayer {
    cl_mem d_FilterBuf;
    cl_mem d_FeatMapBuf;
    cl_mem d_DeltaBuf;
    ConvLayer h_Conv;
    int InputDim;
    int FilterDim;
    int Padding;
    int FeatmapDim;
} cl_ConvLayer;

typedef struct cl_PoolLayer {
    cl_mem d_PoolBuf;
    cl_mem d_UnpoolBuf;
    cl_mem d_IndexBuf;
    ConvLayer h_Conv;
    int FeatmapDim;
    int FilterNum;
    int PoolDim;
} cl_PoolLayer;


class MiniVGG {
public:
    ~MiniVGG();

    void createMiniVGG(std::vector<int> &padding_per_layer, std::vector<int> &num_filters_per_layer, std::vector<int> &filtdim_per_layer, int inpdim, std::vector<cl_int> &newNetVec);

    void train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets,
               std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches);
    void forward();
    void backward();
    void trainingAccuracy(std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int num);

private:
    cl_ConvLayer convLayer1;
    cl_ConvLayer convLayer2;
    cl_ConvLayer convLayer3;
    cl_ConvLayer convLayer4;
    cl_PoolLayer poolLayer1;
    cl_PoolLayer poolLayer2;


    std::vector<ConvLayer> convLayers; // 多層卷積
    std::vector<cl_mem> d_FiltersBuffers;
    std::vector<cl_mem> d_FeatMapBuffers;
    std::vector<cl_mem> d_PoolBuffers;
    std::vector<cl_mem> d_PoolIndexBuffers;

    std::vector<int> featmapdims;
    std::vector<int> pooldims;
    std::vector<int> inputdims;
    std::vector<int> paddings;

    cl_mem d_InputBuffer;
    cl_mem d_targetBuffer;
    cl_mem d_deltasBuffer;
    cl_mem d_rotatedImgBuffer;

    std::vector<cl_mem> d_layersBuffers;

    // std::vector<cl_kernel> convKerns;
    // std::vector<cl_kernel> poolKerns;
    // std::vector<cl_kernel> backpropKerns;

    cl_kernel ConvKern;
    cl_kernel BackKern;
    cl_kernel DeltaKern;

    cl_kernel PoolKern;
    cl_kernel UnpoolKern;

    cl_kernel cnnToFcnnKern;
    cl_kernel compoutKern;
    cl_kernel backpropoutKern;
    cl_kernel backprophidKern;
    cl_kernel softmaxKern;
    cl_kernel deltasKern;
    cl_kernel rotate180Kern;

    std::vector<int> h_netVec;
    std::vector<Layer> h_layers;

    float init_lr = 0.001;
    float lr = 0.001;
    float decay_rate = 0.0001;
    int softflag = 1;
    cl_int err;

    std::unique_ptr<cl_ConvLayer> createConv2d(int* InputDim, int FilterNum, int FilterDim, int Padding);
    std::unique_ptr<cl_PoolLayer> createPool2d(int* InputDim, int FilterNum);

    cl_mem computeConvolution(cl_ConvLayer* pLayer, cl_mem InputBuffer);
    void computeBackpropagation(cl_ConvLayer* pLayer, cl_mem PrevBuffer, cl_mem NextBuffer);
    void computeDelta(cl_ConvLayer* Curr, cl_ConvLayer* Next);
    cl_mem pooling(cl_PoolLayer* pLayer, cl_mem InputBuffer);
    void unpooling(cl_PoolLayer* pLayer, cl_mem InputBuffer);
    void cnntoFcnn(int PoolDim, int FilterNum, cl_mem InputBuffer);
    void computeOutputofNN();
    void calculateError(std::vector<float> desiredout);
};

#endif // MINI_VGG_CONVNN_H
