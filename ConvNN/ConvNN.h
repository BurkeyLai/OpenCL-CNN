#pragma once



#ifndef THESIS_CONVNN_H
#define THESIS_CONVNN_H



#include "include.h"
#include "Layer.h"

class ConvNN {


public:

	// 添加析構函數
	~ConvNN();

	void createConvNN(int numoffilters, int filtdim, int inpdim, int pad);
	void createFullyConnectedNN(std::vector<cl_int> &newNetVec, bool onlyFCNN, int inpdim);


	void train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches);
	void forward(std::vector<float> &input);

	void forwardFCNN(std::vector<float> &input);
	void trainFCNN(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches);


	void trainingAccuracy(std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int num, bool onlyfcnn);

	void calculateError(std::vector<float> desiredout);

	float lr = 0.001;

	int softflag = 1;


private:
	

	///cnn
	//cl::Kernel convKern;
	//cl::Kernel  poolKern;
	//cl::Kernel  reluKern;
	//cl::Kernel  deltasKern;
	//cl::Kernel  backpropcnnKern;
	cl_kernel convKern;
	cl_kernel poolKern;
	cl_kernel reluKern;
	cl_kernel deltasKern;
	cl_kernel backpropcnnKern;


	//cl::Buffer d_InputBuffer;
	//cl::Buffer d_FiltersBuffer;
	//cl::Buffer d_FeatMapBuffer;
	//cl::Buffer d_PoolBuffer;
	//cl::Buffer d_PoolIndexBuffer;
	//cl::Buffer d_targetBuffer;
	//cl::Buffer d_deltasBuffer;
	//cl::Buffer d_rotatedImgBuffer;

	cl_mem d_InputBuffer;
	cl_mem d_FiltersBuffer;
	cl_mem d_FeatMapBuffer;
	cl_mem d_PoolBuffer;
	cl_mem d_PoolIndexBuffer;
	cl_mem d_targetBuffer;
	cl_mem d_deltasBuffer;
	cl_mem d_rotatedImgBuffer;


	ConvLayer convLayer;
	int filterdim;
	int pooldim;
	int featmapdim;
	int inputdim;
	int padding;

	void computeConvolution();
	void pooling();
	void cnntoFcnn();

	///fcnn
	//cl::Kernel compoutKern;
	//cl::Kernel backpropoutKern;
	//cl::Kernel bakckprophidKern;
	//cl::Kernel cnnToFcnnKern;
	//cl::Kernel rotate180Kern;
	//cl::Kernel  softmaxKern;;

	cl_kernel compoutKern;
	cl_kernel backpropoutKern;
	cl_kernel backprophidKern;
	cl_kernel cnnToFcnnKern;
	cl_kernel rotate180Kern;
	cl_kernel  softmaxKern;;


	std::vector<int> h_netVec;
	std::vector<Layer> h_layers;
	//std::vector<cl::Buffer> d_layersBuffers;
	std::vector<cl_mem> d_layersBuffers;


	void computeOutputofNN();


	

	cl_int err;




};


#endif //THESIS_CONVNN_H
