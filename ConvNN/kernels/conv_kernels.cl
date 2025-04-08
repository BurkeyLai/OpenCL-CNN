__kernel void convolve(
    global float* image, 
    global Filter* filters, 
    global float * featMap, 
    int filterWidth, 
    int inWidth, 
    int featmapdim)
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

    float sum = 0;
    for (int r = 0; r < filterWidth; r++){
        for (int c = 0; c < filterWidth; c++){
            sum += filters[z].weights[c + filterWidth * r] * image[(xIn + c) + inWidth * (yIn + r)];
            // sum+= filters[z].weights[(filterWidth-c)+ filterWidth*(filterWidth-r)]*image[(xIn+c)+inWidth *(yIn+r)];
        }
    }

    sum += filters[z].bias;
    //featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=sum/(filterWidth*filterWidth);

    //featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=sigmoid(sum);     
    switch(actflag){
        case 0: featMap[(xIn + yIn * featmapdim + z * featmapdim * featmapdim)] = sigmoid(sum);break;
        case 1: featMap[(xIn + yIn * featmapdim + z * featmapdim * featmapdim)] = mtanh(sum);break;
        case 2: featMap[(xIn + yIn * featmapdim + z * featmapdim * featmapdim)] = relu(sum);break;
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

    float max = 0;
	int index = 0;
    for (int r = 0; r < 2; r++){
        for (int c = 0; c < 2; c++){
            if(prevfeatMap[(yIn + c) * Width * z + (xIn + r)] > max){
                max = prevfeatMap[(yIn + c) * Width * z + (xIn + r)];
                index = c * 2 + r;
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
            if((c * 2 + r) == indexes[i])
                deltas[(2 * xIn + r) + (2 * yIn + c) * dim + z * dim * dim] = delta;      
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

    destin[xIn + dim * yIn] = source[(dim - xIn) + dim * (dim - yIn)];
}

__kernel void backpropcnn(
    global float* featMap,
    global float* deltas,
    global Filter* filters,
    int featmapdim,
    int imagedim,
    int filterdim,
    float a,
    global float* Image)
{
    const int xIn = get_global_id(0);
    const int yIn = get_global_id(1);
    const int z = get_global_id(2);

    float sum = 0;
    for (int r = 0; r < featmapdim; r++){
        for (int c = 0; c < featmapdim; c++){
            sum += deltas[(c + r * featmapdim + z * featmapdim * featmapdim)] * Image[(xIn + r) + imagedim * (yIn + c)];
        }
    }

    filters[z].weights[(xIn + filterdim * yIn)] -= a * sum;///(featmapdim*featmapdim) ;

    // filters[0].bias+=sum;///check this
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
    nodes[(xIn + yIn * inputsize + z * inputsize * inputsize)].output = poolMap[(xIn + yIn * inputsize + z * inputsize * inputsize)];
}


