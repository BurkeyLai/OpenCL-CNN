#include "Layer.h"


float getRandom(float lowerbound, float upperbound)
{
	float f = (float)rand() / RAND_MAX;
	f = lowerbound + f * (upperbound - lowerbound);
	return f;
}

float xavier_init(int fan_in, int fan_out) {
    float limit = sqrt(6.0f / (fan_in + fan_out));
    return ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
}

float he_init(int fan_in) {
    float stddev = sqrt(2.0f / fan_in);
    return stddev * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
}

///Creates a hidden or output layer
Layer* layer(int numberOfNodes, int numberOfWeights)
{
	Layer* hidlayer = new Layer();
	hidlayer->numOfNodes = numberOfNodes;

	for (int i = 0; i != numberOfNodes; ++i)
	{
		Node node;
		node.numberOfWeights = numberOfWeights;


		node.output = 0.0;
		for (int j = 0; j < numberOfWeights; j++)
			// node.weights[j] = getRandom(-0.2, 0.2);
			// node.weights[j] = xavier_init(numberOfNodes, numberOfWeights);
			node.weights[j] = he_init(numberOfWeights);

		hidlayer->nodes[i] = node;
	}
	return hidlayer;
}

ConvLayer* convlayer(int numberOfFilters, int filtdim)
{
	ConvLayer* layer = new ConvLayer();
	layer->numOfFilters = numberOfFilters;

	for (int i = 0; i != numberOfFilters; ++i)
	{
		Filter filter;
		for (int j = 0; j != filtdim; ++j) {
			for (int k = 0; k != filtdim; ++k)
				filter.weights[k * filtdim + j] = getRandom(-0.2, 0.2);
		}
		filter.bias = 0.1;//getRandom(-0.3, 0.3);
		layer->filters[i] = filter;
	}
	return layer;
}

// 添加釋放 Layer 記憶體的函數
void releaseLayer(Layer* layer) {
    if (layer) {
        delete layer;
    }
}

// 添加釋放 ConvLayer 記憶體的函數
void releaseConvLayer(ConvLayer* layer) {
    if (layer) {
        delete layer;
    }
}