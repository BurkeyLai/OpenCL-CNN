// MiniVGG.cpp
#include <memory> // for std::unique_ptr
#include "MiniVGG.h"
#include "OpenCL.h"
#include "err_code.h"

float calculateLoss(const std::vector<float>& output, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        loss += -target[i] * log(output[i] + 1e-7);
    }
    return loss;
}

void debugFirstLayer(Layer &layer) {
	std::cout << "==== Debug First Layer ====" << std::endl;
	for (int i = 0; i < std::min(5, layer.numOfNodes); i++) {
		Node &node = layer.nodes[i];
		std::cout << "Node " << i << ": numWeights = " << node.numberOfWeights << ", first 5 weights = ";
		for (int j = 0; j < std::min(5, node.numberOfWeights); j++) {
			std::cout << node.weights[j] << " ";
		}
		std::cout << std::endl;
	}
}

MiniVGG::~MiniVGG() {
    for (auto buf : d_layersBuffers) clReleaseMemObject(buf);

    clReleaseMemObject(d_InputBuffer);
    clReleaseMemObject(d_targetBuffer);
    clReleaseMemObject(d_deltasBuffer);
    clReleaseMemObject(d_rotatedImgBuffer);

    // for (auto &layer : convLayers) {
    //     clReleaseMemObject(layer->d_FilterBuf);
    //     clReleaseMemObject(layer->d_FeatMapBuf);
    //     clReleaseMemObject(layer->d_DeltaBuf);
    //     clReleaseKernel(layer->ConvKern);
    //     delete layer;
    // }

    // for (auto &pool : poolLayers) {
    //     clReleaseMemObject(pool->d_PoolBuf);
    //     clReleaseMemObject(pool->d_PoolIndex);
    //     clReleaseKernel(pool->PoolKern);
    //     delete pool;
    // }

    clReleaseKernel(cnnToFcnnKern);
    clReleaseKernel(compoutKern);
    clReleaseKernel(backpropoutKern);
    clReleaseKernel(backprophidKern);
    clReleaseKernel(softmaxKern);
    clReleaseKernel(deltasKern);
    clReleaseKernel(rotate180Kern);
}

std::unique_ptr<cl_ConvLayer> MiniVGG::createConv2d(int* InputDim, int FilterNum, int FilterDim, int Padding)
{
    int featmapdim = *InputDim + 2 * Padding - FilterDim + 1;

    std::unique_ptr<cl_ConvLayer> layer = std::make_unique<cl_ConvLayer>();
    ConvLayer* temp = convlayer(FilterNum, FilterDim);
    layer->h_Conv = *temp;
    releaseConvLayer(temp);

    layer->InputDim = *InputDim;
    layer->FilterDim = FilterDim;
    layer->Padding = Padding;
    layer->FeatmapDim = featmapdim;

    layer->d_FilterBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Filter) * FilterNum, NULL, &err);
    clEnqueueWriteBuffer(OpenCL::clqueue, layer->d_FilterBuf, CL_TRUE, 0, sizeof(Filter) * FilterNum, layer->h_Conv.filters, 0, NULL, NULL);
    layer->d_FeatMapBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * featmapdim * featmapdim * FilterNum, NULL, &err);
    layer->d_DeltaBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * featmapdim * featmapdim * FilterNum, NULL, &err);

    *InputDim = featmapdim;

    return layer;
}

std::unique_ptr<cl_PoolLayer> MiniVGG::createPool2d(int* InputDim, int FilterNum)
{
    int pooldim = ((*InputDim - 2) / 2) + 1;

    std::unique_ptr<cl_PoolLayer> layer = std::make_unique<cl_PoolLayer>();

    layer->FeatmapDim = *InputDim;
    layer->FilterNum = FilterNum;
    layer->PoolDim = pooldim;
    layer->d_PoolBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * pooldim * pooldim * FilterNum, NULL, &err);
    layer->d_UnpoolBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * (*InputDim) * (*InputDim) * FilterNum, NULL, &err);
    layer->d_IndexBuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(int) * pooldim * pooldim * FilterNum, NULL, &err);

    *InputDim = pooldim;

    return layer;
}

void MiniVGG::createMiniVGG(std::vector<int> &padding_per_layer, std::vector<int> &num_filters_per_layer, std::vector<int> &filtdim_per_layer, int inpdim, std::vector<cl_int> &newNetVec) {
    int inputdim = inpdim, featmapdim = 0, filternum = 0;
    cl_int err;

    d_InputBuffer = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * inpdim * inpdim, NULL, &err);
    checkError(err, "Input buffer alloc");

    convLayer1 = *createConv2d(&inputdim, 64, 3, 1);
    convLayer2 = *createConv2d(&inputdim, 64, 3, 1);
    poolLayer1 = *createPool2d(&inputdim, 64);
    convLayer3 = *createConv2d(&inputdim, 128, 3, 1);
    convLayer4 = *createConv2d(&inputdim, 128, 3, 1);
    featmapdim = inputdim;
    poolLayer2 = *createPool2d(&inputdim, 128);
    filternum = 128;

    std::vector<float> del(featmapdim * featmapdim * filternum, 0.0);
    d_deltasBuffer = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * featmapdim * featmapdim * filternum, NULL, &err);
	err = clEnqueueWriteBuffer(OpenCL::clqueue, d_deltasBuffer, CL_TRUE, 0, sizeof(float) * featmapdim * featmapdim * filternum, del.data(), 0, NULL, NULL);
	checkError(err, "Finding platforms");

	d_rotatedImgBuffer = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * featmapdim * featmapdim, NULL, &err);

    // Create Fully Connected Layers
    {
        h_netVec = {inputdim * inputdim * filternum, 256, 10}; // 128 * 8 * 8, 256, 10
        // h_netVec = {inputdim * inputdim * filternum, 10};
#if 0
        Layer *inputLayer = layer(h_netVec[0], 0);
        h_layers.push_back(*inputLayer);
        releaseLayer(inputLayer);

        for (unsigned int i = 1; i < h_netVec.size(); i++)
        {
            Layer *hidlayer = layer(h_netVec[i], h_netVec[i - 1]);
            h_layers.push_back(*hidlayer);
            releaseLayer(hidlayer);
        }
#else
        Layer *inputLayer = layer(h_netVec[0], h_netVec[1]);
        h_layers.push_back(*inputLayer);
        releaseLayer(inputLayer);

        // 這段是正確的，因為你寫法是：layer(當前層節點數, 下一層節點數)
        for (unsigned int i = 1; i < h_netVec.size(); i++) {
            if (i + 1 < h_netVec.size()) {
                Layer *hidlayer = layer(h_netVec[i], h_netVec[i + 1]);
                h_layers.push_back(*hidlayer);
                releaseLayer(hidlayer);
            } else {
                // 最後一層 output，不需要再連到下一層
                Layer *outputLayer = layer(h_netVec[i], 0);
                h_layers.push_back(*outputLayer);
                releaseLayer(outputLayer);
            }
        }
#endif
        cl_mem  tempbuf;
        tempbuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node) * h_layers[0].numOfNodes, NULL, &err);
        err = clEnqueueWriteBuffer(OpenCL::clqueue, tempbuf, CL_TRUE, 0, sizeof(Node) * h_layers[0].numOfNodes, h_layers[0].nodes, 0, NULL, NULL);
        d_layersBuffers.push_back(tempbuf);

        for (int i = 1; i < h_layers.size(); i++) {
            tempbuf = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node) * h_layers[i].numOfNodes, NULL, &err);
            err = clEnqueueWriteBuffer(OpenCL::clqueue, tempbuf, CL_TRUE, 0, sizeof(Node) * h_layers[i].numOfNodes, h_layers[i].nodes, 0, NULL, NULL);
            d_layersBuffers.push_back(tempbuf);
        }

        d_targetBuffer = clCreateBuffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float) * h_netVec.back(), NULL, &err);
        checkError(err, "target buffer alloc");
    }

    ConvKern = clCreateKernel(OpenCL::clprogram, "convolve", &err);
    checkError(err, "Kernel convolve");
    BackKern = clCreateKernel(OpenCL::clprogram, "backpropcnn", &err);
    checkError(err, "Kernel backpropcnn");
    DeltaKern = clCreateKernel(OpenCL::clprogram, "backpropdelta", &err);
    checkError(err, "Kernel backpropdelta");

    PoolKern = clCreateKernel(OpenCL::clprogram, "pooling", &err);
    checkError(err, "Kernel pooling");
    UnpoolKern = clCreateKernel(OpenCL::clprogram, "unpool_deltas", &err);
    checkError(err, "Kernel unpool_deltas");

    cnnToFcnnKern   = clCreateKernel(OpenCL::clprogram, "cnntoFcnn", &err);
    compoutKern     = clCreateKernel(OpenCL::clprogram, "compout", &err);
    backpropoutKern = clCreateKernel(OpenCL::clprogram, "backpropout", &err);
    backprophidKern = clCreateKernel(OpenCL::clprogram, "backprophid", &err);
    softmaxKern     = clCreateKernel(OpenCL::clprogram, "softmax", &err);
    deltasKern      = clCreateKernel(OpenCL::clprogram, "deltas", &err);
    rotate180Kern   = clCreateKernel(OpenCL::clprogram, "rotatemat", &err);
}

cl_mem MiniVGG::computeConvolution(cl_ConvLayer* pLayer, cl_mem InputBuffer)
{
    cl_kernel conv = ConvKern;
    int num_filters = pLayer->h_Conv.numOfFilters;
    int filterdim = pLayer->FilterDim;
    int inputdim = pLayer->InputDim;
    int padding = pLayer->Padding;
    int featmapdim = pLayer->FeatmapDim;

    clSetKernelArg(conv, 0, sizeof(cl_mem), &InputBuffer);
    clSetKernelArg(conv, 1, sizeof(cl_mem), &(pLayer->d_FilterBuf));
    clSetKernelArg(conv, 2, sizeof(cl_mem), &(pLayer->d_FeatMapBuf));
    clSetKernelArg(conv, 3, sizeof(int), &filterdim);
    clSetKernelArg(conv, 4, sizeof(int), &inputdim);
    clSetKernelArg(conv, 5, sizeof(int), &featmapdim);
    clSetKernelArg(conv, 6, sizeof(int), &padding);

    size_t global_conv[3] = { (size_t)featmapdim, (size_t)featmapdim, (size_t)num_filters };
    clEnqueueNDRangeKernel(OpenCL::clqueue, conv, 3, NULL, global_conv, NULL, 0, NULL, NULL);

    return pLayer->d_FeatMapBuf;
}

void MiniVGG::computeBackpropagation(cl_ConvLayer* pLayer, cl_mem PrevBuffer, cl_mem NextBuffer)
{
    cl_kernel backprop = BackKern;
    int filterdim = pLayer->FilterDim;
    int featmapdim = pLayer->FeatmapDim;
    int num_filters = pLayer->h_Conv.numOfFilters;
    int inputdim = pLayer->InputDim;
    int padding = pLayer->Padding;

    err = clSetKernelArg(backprop, 0, sizeof(cl_mem), &(pLayer->d_FeatMapBuf));
    err = clSetKernelArg(backprop, 1, sizeof(cl_mem), &NextBuffer);
    err = clSetKernelArg(backprop, 2, sizeof(cl_mem), &(pLayer->d_FilterBuf));
    err = clSetKernelArg(backprop, 3, sizeof(int), &featmapdim);
    err = clSetKernelArg(backprop, 4, sizeof(int), &inputdim);
    err = clSetKernelArg(backprop, 5, sizeof(int), &filterdim);
    err = clSetKernelArg(backprop, 6, sizeof(float), &lr);
    err = clSetKernelArg(backprop, 7, sizeof(cl_mem), &PrevBuffer);
    size_t global_backpropcnn_size[3] = {(size_t)filterdim, (size_t)filterdim, (size_t)num_filters};
    err = clEnqueueNDRangeKernel(OpenCL::clqueue, backprop, 3, NULL, 
        global_backpropcnn_size, NULL, 0, NULL, NULL);

    // Store Delta
    // size_t size = sizeof(float) * featmapdim * featmapdim * num_filters;
    // clEnqueueCopyBuffer(OpenCL::clqueue, NextBuffer, pLayer->d_DeltaBuf, 0, 0, size, 0, NULL, NULL);
}

void MiniVGG::computeDelta(cl_ConvLayer* Curr, cl_ConvLayer* Next)
{
    cl_kernel delta = DeltaKern;  // 指向你之前建好的 backpropDelta kernel

    int filter_width = Next->FilterDim;
    int in_width = Curr->FeatmapDim;
    int next_num_filters = Next->h_Conv.numOfFilters;
    int padding = Next->Padding;

    clSetKernelArg(delta, 0, sizeof(cl_mem), &Next->d_DeltaBuf);
    clSetKernelArg(delta, 1, sizeof(cl_mem), &Next->d_FilterBuf);
    clSetKernelArg(delta, 2, sizeof(cl_mem), &Curr->d_FeatMapBuf);
    clSetKernelArg(delta, 3, sizeof(cl_mem), &Curr->d_DeltaBuf);
    clSetKernelArg(delta, 4, sizeof(int), &filter_width);
    clSetKernelArg(delta, 5, sizeof(int), &in_width);
    clSetKernelArg(delta, 6, sizeof(int), &next_num_filters);
    clSetKernelArg(delta, 7, sizeof(int), &padding);

    size_t global[3] = { (size_t)in_width, (size_t)in_width, (size_t)Curr->h_Conv.numOfFilters };
    clEnqueueNDRangeKernel(OpenCL::clqueue, delta, 3, NULL, global, NULL, 0, NULL, NULL);
}

cl_mem MiniVGG::pooling(cl_PoolLayer* pLayer, cl_mem InputBuffer)
{
    cl_kernel pool = PoolKern;
    int featmapdim = pLayer->FeatmapDim;
    int pooldim = pLayer->PoolDim;
    int num_filters = pLayer->FilterNum;

    clSetKernelArg(pool, 0, sizeof(cl_mem), &InputBuffer);
    clSetKernelArg(pool, 1, sizeof(cl_mem), &(pLayer->d_PoolBuf));
    clSetKernelArg(pool, 2, sizeof(cl_mem), &(pLayer->d_IndexBuf));
    clSetKernelArg(pool, 3, sizeof(int), &featmapdim);
    clSetKernelArg(pool, 4, sizeof(int), &pooldim);

    size_t global[3] = { (size_t)pooldim, (size_t)pooldim, (size_t)num_filters };
    clEnqueueNDRangeKernel(OpenCL::clqueue, pool, 3, NULL, global, NULL, 0, NULL, NULL);

    return pLayer->d_PoolBuf;
}

void MiniVGG::unpooling(cl_PoolLayer* pLayer, cl_mem InputBuffer)
{
    cl_kernel unpool = UnpoolKern;
    int featmapdim = pLayer->FeatmapDim;
    int pooldim = pLayer->PoolDim;
    int num_filters = pLayer->FilterNum;

    clSetKernelArg(unpool, 0, sizeof(cl_mem), &InputBuffer);
    clSetKernelArg(unpool, 1, sizeof(cl_mem), &(pLayer->d_IndexBuf));
    clSetKernelArg(unpool, 2, sizeof(cl_mem), &(pLayer->d_UnpoolBuf));
    clSetKernelArg(unpool, 3, sizeof(int), &pooldim);
    clSetKernelArg(unpool, 4, sizeof(int), &featmapdim);

    size_t global[3] = { (size_t)pooldim, (size_t)pooldim, (size_t)num_filters };
    clEnqueueNDRangeKernel(OpenCL::clqueue, unpool, 3, NULL, global, NULL, 0, NULL, NULL);
}

void MiniVGG::cnntoFcnn(int PoolDim, int FilterNum, cl_mem InputBuffer)
{
    clSetKernelArg(cnnToFcnnKern, 0, sizeof(cl_mem), &InputBuffer);
    clSetKernelArg(cnnToFcnnKern, 1, sizeof(cl_mem), &d_layersBuffers[0]);
    clSetKernelArg(cnnToFcnnKern, 2, sizeof(int), &PoolDim);
    size_t global[3] = { (size_t)PoolDim, (size_t)PoolDim, (size_t)FilterNum };
    clEnqueueNDRangeKernel(OpenCL::clqueue, cnnToFcnnKern, 3, NULL, global, NULL, 0, NULL, NULL);
}

void MiniVGG::computeOutputofNN()
{
    for (int i = 1; i < h_layers.size(); i++) {
        int sf = (i == h_layers.size() - 1 && softflag == 1) ? 1 : 0;
        clSetKernelArg(compoutKern, 0, sizeof(cl_mem), &d_layersBuffers[i]);
        clSetKernelArg(compoutKern, 1, sizeof(cl_mem), &d_layersBuffers[i - 1]);
        clSetKernelArg(compoutKern, 2, sizeof(int), &sf);
        size_t global[1] = { (size_t)h_netVec[i] };
        clEnqueueNDRangeKernel(OpenCL::clqueue, compoutKern, 1, NULL, global, NULL, 0, NULL, NULL);

        if (sf) {
            clSetKernelArg(softmaxKern, 0, sizeof(cl_mem), &d_layersBuffers[i]);
            clSetKernelArg(softmaxKern, 1, sizeof(int), &h_layers[i].numOfNodes);
            clEnqueueNDRangeKernel(OpenCL::clqueue, softmaxKern, 1, NULL, global, NULL, 0, NULL, NULL);
        }
    }
}

void MiniVGG::train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets ,int epoches)
{
	int i = 0;

    // debugFirstLayer(h_layers[0]);

	for (int e = 0; e<epoches; e++) {
		i = e % inputs.size();
        // printf("epoch %d\n", e);
        lr = init_lr * (1.0f / (1.0f + decay_rate * e));

		err = clEnqueueWriteBuffer(OpenCL::clqueue, d_targetBuffer, CL_TRUE, 0, sizeof(float) * h_netVec.back(), targets[i].data(), 0, NULL, NULL);
		checkError(err, "EnqueueWriteBuffer");

        err = clEnqueueWriteBuffer(OpenCL::clqueue, d_InputBuffer, CL_TRUE, 0, sizeof(float) * convLayer1.InputDim * convLayer1.InputDim, inputs[i].data(), 0, NULL, NULL);
        checkError(err, "EnqueueWriteBuffer");

		// forward
        forward();
        // printf("forward done\n");

		if (e % 1000 == 0) {
			std::cout << e << std::endl;
		}
#if 0
        std::vector<Node> input_layer_nodes(h_layers[0].numOfNodes);
        clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers[0], CL_TRUE, 0,
            sizeof(Node) * input_layer_nodes.size(), input_layer_nodes.data(), 0, NULL, NULL);

        std::cout << "CNN→FCNN輸入層 output（前 10 個）: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << input_layer_nodes[i].output << " ";
        }
        std::cout << std::endl;
#endif
#if 0
        std::vector<Node> outnodes(h_layers.back().numOfNodes);
        clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers.back(), CL_TRUE, 0,
            sizeof(Node) * outnodes.size(), outnodes.data(), 0, NULL, NULL);

        std::cout << "Softmax output: ";
        for (auto &n : outnodes)
            std::cout << n.output << " ";
        std::cout << std::endl;
#endif

		calculateError(targets[i]);
        // printf("calculateError done\n");

        if (e % 100 == 0) {
            // 1. 確保大小正確
            size_t output_size = h_layers.back().numOfNodes;
            std::vector<Node> temp_nodes(output_size);
            std::vector<float> current_output(output_size);

            // 2. 先讀取完整的 Node 結構
            cl_int err = clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers.back(), CL_TRUE, 0, sizeof(Node) * output_size, temp_nodes.data(), 0, NULL, NULL);

            // 3. 檢查錯誤
            if (err != CL_SUCCESS) {
                std::cerr << "Error reading output buffer: " << err << std::endl;
                exit(1);
            }

            // 4. 從 Node 結構中提取 output 值
            for (size_t i = 0; i < output_size; ++i) {
                current_output[i] = temp_nodes[i].output;
            }

            // 5. 計算 loss
            float loss = calculateLoss(current_output, targets[i]);
            std::cout << "Epoch " << e << " Loss: " << loss << std::endl;

            // 6. 同步 OpenCL 命令佇列
            clFinish(OpenCL::clqueue);
        }

		// backward
		backward();
        // printf("backward done\n");

        if (e % 10000 == 0 && e!=0) {
            trainingAccuracy(testinputs, testtargets, 2000);
        }
	}
}

void MiniVGG::forward() {
    cl_mem output = computeConvolution(&convLayer1, d_InputBuffer);
    clFinish(OpenCL::clqueue);
    // printf("conv1 done\n");
    output = computeConvolution(&convLayer2, output);
    clFinish(OpenCL::clqueue);
    // printf("conv2 done\n");
    output = pooling(&poolLayer1, output);
    clFinish(OpenCL::clqueue);
    // printf("pool1 done\n");
    output = computeConvolution(&convLayer3, output);
    clFinish(OpenCL::clqueue);
    // printf("conv3 done\n");
    output = computeConvolution(&convLayer4, output);
    clFinish(OpenCL::clqueue);
    // printf("conv4 done\n");
    output = pooling(&poolLayer2, output);
    clFinish(OpenCL::clqueue);
    // printf("pool2 done\n");
    cnntoFcnn(poolLayer2.PoolDim, poolLayer2.FilterNum, output);
    clFinish(OpenCL::clqueue);
    // printf("cnntoFcnn done\n");
    computeOutputofNN();
    clFinish(OpenCL::clqueue);
    // printf("computeOutputofNN done\n");
}

void MiniVGG::backward() {
    for (int l = h_layers.size() - 1; l>0; l--) {
        if (l == h_layers.size() - 1) {
            err = clSetKernelArg(backpropoutKern, 0, sizeof(cl_mem), &d_layersBuffers[l]);
            err = clSetKernelArg(backpropoutKern, 1, sizeof(cl_mem), &d_layersBuffers[l - 1]);
            err = clSetKernelArg(backpropoutKern, 2, sizeof(cl_mem), &d_targetBuffer);
            err = clSetKernelArg(backpropoutKern, 3, sizeof(float), &lr);
            err = clSetKernelArg(backpropoutKern, 4, sizeof(int), &softflag);

            size_t global_work_size[1] = {(size_t)h_netVec[l]};
            err = clEnqueueNDRangeKernel(OpenCL::clqueue, backpropoutKern, 1, NULL, 
                global_work_size, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                std::cerr << "Enqueue backpropout failed, error code: " << err << std::endl;
            }
#if 0
            std::vector<Node> prev_layer_nodes(h_layers[0].numOfNodes);
            clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers[0], CL_TRUE, 0,
                sizeof(Node) * prev_layer_nodes.size(), prev_layer_nodes.data(), 0, NULL, NULL);

            std::cout << "Prev Layer outputs (first 5): ";
            for (int i = 0; i < 5; i++) {
                std::cout << prev_layer_nodes[i].output << " ";
            }
            std::cout << std::endl;
#endif
#if 0
            std::vector<Node> outnodes(h_layers.back().numOfNodes);
            clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers.back(), CL_TRUE, 0,
                sizeof(Node) * outnodes.size(), outnodes.data(), 0, NULL, NULL);

            std::cout << "FCNN Output Layer Delta（前 10 個）: ";
            for (int i = 0; i < 10; ++i)
                std::cout << outnodes[i].delta << " ";
            std::cout << std::endl;
#endif
        } else {
            err = clSetKernelArg(backprophidKern, 0, sizeof(cl_mem), &d_layersBuffers[l]);
            err = clSetKernelArg(backprophidKern, 1, sizeof(cl_mem), &d_layersBuffers[l - 1]);
            err = clSetKernelArg(backprophidKern, 2, sizeof(cl_mem), &d_layersBuffers[l + 1]);
            err = clSetKernelArg(backprophidKern, 3, sizeof(cl_int), &h_netVec[l+1]);
            err = clSetKernelArg(backprophidKern, 4, sizeof(float), &lr);

            size_t global_work_size[1] = {(size_t)h_netVec[l]};
            err = clEnqueueNDRangeKernel(OpenCL::clqueue, backprophidKern, 1, NULL, 
                global_work_size, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                std::cerr << "Enqueue backprophid failed, error code: " << err << std::endl;
            }
        }
    }
#if 0
    std::vector<Node> fc_input_nodes(h_layers[0].numOfNodes);
    clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers[0], CL_TRUE, 0,
        sizeof(Node) * fc_input_nodes.size(), fc_input_nodes.data(), 0, NULL, NULL);

    static std::vector<float> prev_weights(5, 0.0f);  // 初始為 0
    for (int i = 0; i < 5; i++) {
        float now = fc_input_nodes[i].weights[0];
        float delta = now - prev_weights[i];
        std::cout << "Node[" << i << "].weights[0] Δ: " << delta << std::endl;
        prev_weights[i] = now;
    }
#endif

    // printf("hidden layer done\n");

    int featmapdim = poolLayer2.FeatmapDim;
    int num_filters = poolLayer2.FilterNum;
    int pooldim = poolLayer2.PoolDim;

    err = clSetKernelArg(deltasKern, 0, sizeof(cl_mem), &d_layersBuffers[0]);
    err = clSetKernelArg(deltasKern, 1, sizeof(cl_mem), &d_layersBuffers[1]);
    err = clSetKernelArg(deltasKern, 2, sizeof(cl_mem), &d_deltasBuffer);
    err = clSetKernelArg(deltasKern, 3, sizeof(cl_mem), &poolLayer2.d_IndexBuf);
    err = clSetKernelArg(deltasKern, 4, sizeof(int), &featmapdim);
    err = clSetKernelArg(deltasKern, 5, sizeof(cl_int), &h_netVec[1]);
    err = clSetKernelArg(deltasKern, 6, sizeof(int), &pooldim);
    size_t global_deltas_size[3] = {(size_t)pooldim, (size_t)pooldim, (size_t)num_filters};
    err = clEnqueueNDRangeKernel(OpenCL::clqueue, deltasKern, 3, NULL, 
        global_deltas_size, NULL, 0, NULL, NULL);
    // printf("delta done\n");

    err = clSetKernelArg(rotate180Kern, 0, sizeof(cl_mem), &d_deltasBuffer);
    err = clSetKernelArg(rotate180Kern, 1, sizeof(cl_mem), &d_rotatedImgBuffer);
    err = clSetKernelArg(rotate180Kern, 2, sizeof(int), &featmapdim);
    size_t global_rotate_size[2] = {(size_t)featmapdim, (size_t)featmapdim};
    err = clEnqueueNDRangeKernel(OpenCL::clqueue, rotate180Kern, 2, NULL, 
        global_rotate_size, NULL, 0, NULL, NULL);
    // printf("rotate180 done\n");
    // delta from FCNN → conv4
    computeBackpropagation(&convLayer4, convLayer3.d_FeatMapBuf, d_rotatedImgBuffer);
    // printf("back conv4 done\n");
    // delta from conv4 → conv3
    computeBackpropagation(&convLayer3, poolLayer1.d_PoolBuf, convLayer4.d_DeltaBuf);
    // printf("back conv3 done\n");
    computeDelta(&convLayer3, &convLayer4);
    // printf("delta conv3 done\n");
    unpooling(&poolLayer1, convLayer3.d_DeltaBuf);
    // printf("unpool done\n");
    computeBackpropagation(&convLayer2, convLayer1.d_FeatMapBuf, poolLayer1.d_UnpoolBuf);
    // printf("back conv2 done\n");
    computeDelta(&convLayer2, &convLayer3);
    // printf("delta conv2 done\n");
    computeBackpropagation(&convLayer1, d_InputBuffer, convLayer1.d_DeltaBuf);
    // printf("back conv1 done\n");
    computeDelta(&convLayer1, &convLayer2);
    // printf("delta conv1 done\n");
}

void MiniVGG::calculateError(std::vector<float> desiredout) {
    Node bufdata[10];
	clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers.back(), CL_TRUE, 0, sizeof(Node) * h_layers.back().numOfNodes, bufdata, 0, NULL, NULL);
	float error=0;
}

void MiniVGG::trainingAccuracy(std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int num) {
    float testerrors = 0;

	for (int i = 0; i < num; i++) {
        Node bufdata[10];

        err = clEnqueueWriteBuffer(OpenCL::clqueue, d_InputBuffer, CL_TRUE, 0, sizeof(float) * convLayer1.InputDim * convLayer1.InputDim, testinputs[i].data(), 0, NULL, NULL);
        checkError(err, "EnqueueWriteBuffer");
		forward();
		clEnqueueReadBuffer(OpenCL::clqueue, d_layersBuffers.back(), CL_TRUE, 0, sizeof(Node) * h_layers.back().numOfNodes, bufdata, 0, NULL, NULL);
		// dumpBufferNodes(d_layersBuffers.back(), h_layers.back().numOfNodes);

		//findmax
		float max = 0;
		int maxindex = 0;
		for (int j = 0; j < h_netVec.back(); j++) {
			if (bufdata[j].output > max) {
				max = bufdata[j].output;
				// std::cout << "max:      " << max << std::endl;
				maxindex = j;
				// std::cout << "maxindex: " << maxindex << std::endl;
			}
		}

		if (maxindex ==(int) testtargets[i]) testerrors++;
	}

	std::cout << "Net:" << std::endl;
	std::cout << "Num of filters: " << convLayer1.h_Conv.numOfFilters << std::endl;
	std::cout << "Filter dimension: " << convLayer1.FilterDim << std::endl;

    std::cout << "Num of filters: " << convLayer2.h_Conv.numOfFilters << std::endl;
	std::cout << "Filter dimension: " << convLayer2.FilterDim << std::endl;

    std::cout << "Num of filters: " << convLayer3.h_Conv.numOfFilters << std::endl;
	std::cout << "Filter dimension: " << convLayer3.FilterDim << std::endl;

    std::cout << "Num of filters: " << convLayer4.h_Conv.numOfFilters << std::endl;
	std::cout << "Filter dimension: " << convLayer4.FilterDim << std::endl;

	std::cout << "Fullconnected:";
	for (int i = 0; i < h_netVec.size();i++) {
		std::cout << h_netVec[i] <<" ";
	}
	std::cout << std::endl;
	std::cout << "Learning rate: " << lr << std::endl;

	//std::cout << "    Completed in " << difftime(end ,start) << " seconds" << endl;
	std::cout << "NUMBER OF CORRECT: " << testerrors << " CORRECT RATE: " << 100 * (testerrors / num) << "%" << std::endl;
}

