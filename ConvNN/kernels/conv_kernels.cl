__kernel void convolve(
    global float* image, 
    global Filter* filters, 
    global float * featMap, 
    int filterWidth, 
    int inWidth, 
    int featmapdim,
    int padding)
{
    const int xIn = get_global_id(0);//cols
    const int yIn = get_global_id(1);//rows
	const int z = get_global_id(2);//filters

    // Input Image * Filter Kernel
    // 
    // x11 x12 x13   w11 w12
    // x21 x22 x23 * w21 w22
    // x31 x32 x33
    //
    // H = X * W
    //
    // h11 = w11x11 + w12x12 + w21x21 + w22x22, (xIn, yIn) = (0, 0)
    // h12 = w11x12 + w12x13 + w21x22 + w22x23, (xIn, yIn) = (1, 0)
    // h21 = w11x21 + w12x22 + w21x31 + w22x32, (xIn, yIn) = (0, 1)
    // h22 = w11x22 + w12x23 + w21x32 + w22x33, (xIn, yIn) = (1, 1)
    //
    // xIn 代表 W 在 X 中 map 到的參數，其 column index
    // yIn 代表 W 在 X 中 map 到的參數，其 row index
    if ((xIn + filterWidth) > inWidth || (yIn + filterWidth) > inWidth) return;

    // float sum = 0;
    // for (int r = 0; r < filterWidth; r++){
    //     for (int c = 0; c < filterWidth; c++){
    //         sum += filters[z].weights[c + filterWidth * r] * image[(xIn + c) + inWidth * (yIn + r)];
    //         // sum+= filters[z].weights[(filterWidth-c)+ filterWidth*(filterWidth-r)]*image[(xIn+c)+inWidth *(yIn+r)];
    //     }
    // }

    // 計算輸出特徵圖的大小
    int outWidth = inWidth + 2 * padding - filterWidth + 1;
    // 計算輸入圖像的索引，考慮 padding
    float sum = 0;
    for (int r = 0; r < filterWidth; r++) {
        for (int c = 0; c < filterWidth; c++) {
            // 計算輸入圖像的索引，考慮 padding
            int imgX = xIn + c - padding;
            int imgY = yIn + r - padding;
            
            // 檢查是否在 padding 區域
            if (imgX < 0 || imgX >= inWidth || imgY < 0 || imgY >= inWidth) {
                // 在 padding 區域，使用 0 值（zero padding）
                sum += filters[z].weights[c + filterWidth * r] * 0.0f;
            } else {
                // 在有效區域，使用實際圖像值
                sum += filters[z].weights[c + filterWidth * r] * image[imgX + inWidth * imgY];
            }
        }
    }

    // int featMapIndex = xIn + yIn * featmapdim + z * featmapdim * featmapdim;
    int featMapIndex = (xIn + padding) + (yIn + padding) * featmapdim + z * featmapdim * featmapdim;
    sum += filters[z].bias;
    switch(actflag){
        case 0: featMap[featMapIndex] = sigmoid(sum);break;
        case 1: featMap[featMapIndex] = mtanh(sum);break;
        case 2: featMap[featMapIndex] = relu(sum);break;
    }
}

__kernel void pooling(
    global float* prevfeatMap,
    global float* poolMap,
    global int* indexes,
    int Width,
    int pooldim)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);
    const int z = get_global_id(2);

    float max = -MAXFLOAT; // float max = 0;
	int index = 0;
    for (int r = 0; r < 2; r++){
        for (int c = 0; c < 2; c++){
            if(prevfeatMap[(xIn + c) + Width * (yIn + r) + z * Width * Width] > max){
                max = prevfeatMap[(xIn + c) + Width * (yIn + r) + z * Width * Width];
                index = r * 2 + c;
            }
        }
    }

    poolMap[(xIn + yIn * pooldim + z * pooldim * pooldim)] = max;
    indexes[(xIn + yIn * pooldim + z * pooldim * pooldim)] = index;
}

__kernel void deltas(
    global Node* nodes,
    global Node* nextnodes,
    global float* deltas,
    global int* indexes,
    int dim,
    int nextnumNodes,
    int pooldim)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);
    const int z = get_global_id(2);

	int i = xIn + yIn * pooldim + z * pooldim * pooldim;

    float delta = 0;
    for (int j = 0; j != nextnumNodes; j++)
        delta += nextnodes[j].delta * nextnodes[j].weights[i];

	switch(actflag){
		case 0: delta *= devsigmoid(nodes[i].output);break;
		case 1: delta *= devtanh(nodes[i].output);break;
		case 2: delta *= devrelu(nodes[i].output);break;
	}

    for(int r = 0; r < 2; r++){
        for(int c = 0;c < 2; c++){
            if((r * 2 + c) == indexes[i])
                deltas[(2 * xIn + c) + (2 * yIn + r) * dim + z * dim * dim] = delta;      
        }
    }
}

__kernel void rotatemat(
    global float* source,
    global float* destin,
    int dim)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);

    destin[xIn + dim * yIn] = source[(dim - 1 - xIn) + dim * (dim - 1 - yIn)];
}

__kernel void backpropcnn(
    global float* featMap,
    global float* deltas,
    global Filter* filters,
    int featmapdim,
    int imagedim,
    int filterdim,
    int padding,
    float a,
    global float* Image)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);
    const int z = get_global_id(2);

    // if ((xIn + featmapdim) > imagedim || (yIn + featmapdim) > imagedim) return;
    // float sum = 0;
    // for (int r = 0; r < featmapdim; r++){
    //     for (int c = 0; c < featmapdim; c++){
    //         sum += deltas[(xIn + c) + (yIn + r) * featmapdim + z * featmapdim * featmapdim] * Image[(xIn + c) + imagedim * (yIn + r)];
    //     }
    // }

    float sum = 0.0f;
    for (int r = 0; r < featmapdim; r++) {
        for (int c = 0; c < featmapdim; c++) {
            int imgX = xIn + c - padding;
            int imgY = yIn + r - padding;
            float img_val = 0.0f;
            if (imgX >= 0 && imgX < imagedim && imgY >= 0 && imgY < imagedim) {
                img_val = Image[imgX + imagedim * imgY];
            }
            sum += deltas[(xIn + c) + (yIn + r) * featmapdim + z * featmapdim * featmapdim] * img_val;
        }
    }

    filters[z].weights[(xIn + filterdim * yIn)] -= a * sum;///(featmapdim*featmapdim) ;

    // filters[0].bias+=sum;///check this
}

__kernel void unpool_deltas(
    __global float* pooled_deltas,  // 來自上一層的 delta（pooling 後）
    __global int* pool_indexes,     // pooling 時記下來的 max 位置（0~3）
    __global float* full_deltas,    // 展開到 pooling 前的大小
    int pooldim,                    // pooled delta 空間大小
    int full_dim                    // original feat map 空間大小（= 2*pooldim）
) {
    int x = get_global_id(0);  // pooled x
    int y = get_global_id(1);  // pooled y
    int z = get_global_id(2);  // channel

    int pooled_idx = x + y * pooldim + z * pooldim * pooldim;
    int max_index = pool_indexes[pooled_idx];
    float delta_val = pooled_deltas[pooled_idx];

    // 對應展開回 full_deltas 的哪一格 (2x2)
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            if (r * 2 + c == max_index) {
                int full_x = 2 * x + c;
                int full_y = 2 * y + r;
                int full_idx = full_x + full_y * full_dim + z * full_dim * full_dim;
                full_deltas[full_idx] = delta_val;
            }
        }
    }
}

//__kernel void cnntoFcnn(
//    global float* poolMap,
//    global Node* nodes,
//    int inputsize,
//    int mapindex)
__kernel void cnntoFcnn(
    global float* poolMap,
    global Node* nodes,
    int inputsize)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);
    const int z = get_global_id(2);
    int index = xIn + yIn * inputsize + z * inputsize * inputsize;
    nodes[index].output = poolMap[index];
}

__kernel void backpropdelta(
    __global float* next_deltas,       // delta 來自上一層 (例如 conv3)
    __global Filter* next_filters,     // 上一層的 filters (例如 conv3 的 filter)
    __global float* curr_outputs,      // 當前層的 feature map (activation)
    __global float* curr_deltas,       // 本層要寫入的 delta (conv2.d_DeltaBuf)
    int filter_width,                  // 上一層的 filter 大小
    int in_width,                      // 本層 feature map 的寬度
    int next_num_filters,             // 上一層 filter 數量 (等於 delta channel 數)
    int padding                        // zero padding
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2); // 通道 index = 本層 channel index

    float delta = 0.0f;
    for (int nf = 0; nf < next_num_filters; nf++) {
        for (int r = 0; r < filter_width; r++) {
            for (int s = 0; s < filter_width; s++) {
                int x_n = x - r + padding;
                int y_n = y - s + padding;
                if (x_n >= 0 && x_n < in_width && y_n >= 0 && y_n < in_width) {
                    float w = next_filters[nf].weights[c + r * filter_width + s * filter_width * filter_width];
                    float d = next_deltas[x_n + y_n * in_width + nf * in_width * in_width];
                    delta += w * d;
                }
            }
        }
    }

    // 乘上 activation 對 z 的偏導（activation derivative）
    int index = x + y * in_width + c * in_width * in_width;
    float out = curr_outputs[index];
    switch (actflag) {
        case 0: delta *= devsigmoid(out); break;
        case 1: delta *= devtanh(out); break;
        case 2: delta *= devrelu(out); break;
    }

    curr_deltas[index] = delta;
}

