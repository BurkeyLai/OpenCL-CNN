

__kernel void compout(
	global Node* nodes,
	global Node* prevnodes,
	int softflag)
{
    // const int n = get_global_size(0);
    const int i = get_global_id(0);

    float t = 0;

    for ( int j = 0; j < nodes[i].numberOfWeights; j++)
    	t += nodes[i].weights[j] * prevnodes[j].output;

	t += 0.1; //bias
	
  	if (softflag == 0) {
		switch(actflag) {
			case 0:nodes[i].output = sigmoid(t);break;
			case 1:nodes[i].output = mtanh(t);break;
			case 2:nodes[i].output = relu(t);break;
		}
	} else {
		nodes[i].output = t;
	}
}

__kernel void softmax(
	global Node* nodes,
	int nodesnum){
	
    const int i = get_global_id(0);

	float expsum = 0;
	for (int j = 0; j < nodesnum; j++)
		expsum += exp(nodes[j].output); // Calculates the exp sum of all nodes' output
	// Each node's softmax can be calculated
	nodes[i].output = exp(nodes[i].output) / expsum;
}


__kernel void backprophid(
	global Node* nodes,
	global Node* prevnodes,
	global Node *nextnodes,
	int nextnumNodes,
	float a)
{
    // const int n = get_global_size(0);
    const int i = get_global_id(0);

	// 遍歷下一層的每個節點，將當前節點的誤差項累加為 delta += nextnodes[j].delta * nextnodes[j].weights[i]，即將下一層節點的誤差項乘以對應的權重加到 delta 中。
    float delta = 0;
    for (int j = 0; j != nextnumNodes; j++)
        delta += nextnodes[j].delta * nextnodes[j].weights[i];
        // delta *= nodes[i].output * (1 - nodes[i].output);

	switch(actflag){
		case 0: delta *= devsigmoid(nodes[i].output);break;
		case 1: delta *= devtanh(nodes[i].output);break;
		case 2: delta *= devrelu(nodes[i].output);break;
	}

   	nodes[i].delta = delta;

    for (int j = 0; j != nodes[i].numberOfWeights; j++)
        nodes[i].weights[j] -= a * delta * prevnodes[j].output;
}


__kernel void backpropout(
	global Node* nodes,
	global Node* prevnodes,
	global float* targets,
	float a,
	int softflag)
{
	// const int n = get_global_size(0);
	const int i = get_global_id(0);

	float delta = 0;

	if(softflag == 1){
		delta = nodes[i].output - targets[i];
	} else{
		switch(actflag){ // 每個節點的誤差項 delta
			case 0: delta = (nodes[i].output - targets[i]) * devsigmoid(nodes[i].output);break;
			case 1: delta = (nodes[i].output - targets[i]) * devtanh(nodes[i].output);break;
			case 2: delta = nodes[i].output - targets[i] * devrelu(nodes[i].output);break;
		}
	}

	// 用學習率 a、誤差項 delta 和前一層節點的輸出 prevnodes[j].output 來更新權重
	for (int j = 0; j != nodes[i].numberOfWeights; j++)
		nodes[i].weights[j] -= a * delta * prevnodes[j].output;

	nodes[i].delta = delta;
}



    


