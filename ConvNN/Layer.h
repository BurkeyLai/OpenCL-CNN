#pragma once


#ifndef LAYER_H
#define LAYER_H

#include "include.h"



typedef struct Node {

	int numberOfWeights;
	float weights[10000];
	float output;
	float delta;

}Node;

typedef struct Filter {

	float weights[49];
	float bias;

}Filter;


typedef struct Layer {

	int numOfNodes;
	Node nodes[10000];

}Layer;

typedef struct ConvLayer {

	int numOfFilters;
	Filter filters[128];

}ConvLayer;






Layer* layer(int numberOfNodes, int numberOfWeights);
ConvLayer* convlayer(int numberOfFilters, int filtdim);

// 添加釋放記憶體的函數聲明
void releaseLayer(Layer* layer);
void releaseConvLayer(ConvLayer* layer);





#endif //LAYER_H
