// Thesis.cpp : Defines the entry point for the console application.
//



#include "OpenCL.h"
#include "ConvNN.h"
#include "MiniVGG.h"
#include "util.h"

// 修改 OpenCV 包含路徑，使用相對路徑
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

void read_Mnist(string filename, vector<vector<float>> &vec);
void read_Mnist_Label(string filename, vector<vector<float>> &vec, vector<float> &testtargets, bool testflag);
void printInput(vector<float> &inputs);
void read_CIFAR10(cv::Mat &trainX, cv::Mat &testX, cv::Mat &trainY, cv::Mat &testY);

// 新增CNN執行函數
void runCNN(vector<vector<float>> &inputs, 
            vector<vector<float>> &targets, 
            vector<vector<float>> &testinputs, 
            vector<float> &testtargets,
            util::Timer &timer)
{

    // CNN
    ConvNN m_nn;
    m_nn.createConvNN(7, 7, 32, 0); // num of filters, filterdim, imagedim, padding

    // todo::many filters  3d kernel
    vector<int> netVec;
    netVec = { 169 * 7, 10 }; // {13 * 13 * 7, 10} = {1183, 10}, 13 = pooldim
    m_nn.createFullyConnectedNN(netVec, 0, 32);

    m_nn.train(inputs, targets, testinputs, testtargets, 1000000);

    cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;
}

// 幫我創建另一個檔案，類似 ConvNN.cpp，我要用來建立一個 MiniVGG 模型，模型裡頭有多個
void runMiniVGG(vector<vector<float>> &inputs, 
            	vector<vector<float>> &targets, 
            	vector<vector<float>> &testinputs, 
            	vector<float> &testtargets,
            	util::Timer &timer)
{

    // self.conv_layers = nn.Sequential(
	// 	nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
	// 	nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
	// 	nn.MaxPool2d(2),  # 16x16
	// 	nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
	// 	nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
	// 	nn.MaxPool2d(2),  # 8x8
	// )
	// self.fc_layers = nn.Sequential(
	// 	nn.Flatten(),
	// 	nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
	// 	nn.Linear(256, 10)
	// )
	
	// CNN
    MiniVGG m_nn;
	vector<int> netVec;
	vector<int> padding_per_layer;
	vector<int> num_filters_per_layer;
	vector<int> filtdim_per_layer;
	m_nn.createMiniVGG(padding_per_layer, num_filters_per_layer, filtdim_per_layer, 32, netVec);

    m_nn.train(inputs, targets, testinputs, testtargets, 1000000);

    cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;
}

void runLeNet(vector<vector<float>> &inputs, 
              vector<vector<float>> &targets, 
              vector<vector<float>> &testinputs, 
              vector<float> &testtargets,
              util::Timer &timer)
{

	ConvNN m_nn;
    m_nn.createConvNN(6, 5, 32, 0); // num of filters, filterdim, imagedim, padding
}

int main(void)
{
	uint32_t Ret = 0;

	try {

		Ret = OpenCL::initialize_OpenCL();



		util::Timer timer;

		timer.reset();




		vector<vector<float> > inputs, targets;
		vector<vector<float> > testinputs;
		vector<float> testtargets;

		// {
		//	vector<vector<float> > mnist_inputs;
		//	read_Mnist("mnist/train-images.idx3-ubyte", mnist_inputs);
		//}

		/*//////////////////////////////////
		vector<float> intemp(28 * 28);

		for (int i = 0; i < 28 * 28; i++) {
			intemp.at(i) = 0.5;
		}

		for (int j = 0; j < 10000; j++)
			inputs.push_back(intemp);


		vector<float> temp(10);

		for (int i = 0; i < 1; i++)
			temp.at(i) = 0;
		temp.at(1) = 1;

		for (int j = 0; j < 10000; j++)
			targets.push_back(temp);

		testinputs = inputs;
		for (int i = 0; i < 10000; i++)
			testtargets.push_back(1);
		

		////////////////////////////////////////////////////////*/

		///MNIST
		/*//////////////////////////////////////////////////
		read_Mnist("train-images.idx3-ubyte", inputs);
		read_Mnist_Label("train-labels.idx1-ubyte", targets,testtargets,0);


		cout << "MNIST loaded in: " <<timer.getTimeMilliseconds()/1000.0 <<" s"<<endl;

		timer.reset();
		read_Mnist("t10k-images.idx3-ubyte", testinputs);
		read_Mnist_Label("t10k-labels.idx1-ubyte", targets, testtargets, 1);

		//for (int i = 0; i < 30; i++)
			//cout << " " <<testtargets[i];
		cout << "MNIST test loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;

		//printInput(inputs[54]);

		////////////////////////////////////////////////////*/


		///CIFAR10
		/////////////////////////////////////////////////////////
		cv::Mat trainX, testX;

		cv::Mat trainY, testY;

		// <-     50000     ->
		// img img img ... img 1024
		trainX = cv::Mat::zeros(1024, 50000, CV_32FC1); // 圖片被 resized: 32 * 32 -> 1024 * 1

		testX = cv::Mat::zeros(1024, 10000, CV_32FC1);

		// <-         50000         ->
		// label label label ... label 1
		trainY = cv::Mat::zeros(1, 50000, CV_32FC1);

		testY = cv::Mat::zeros(1, 10000, CV_32FC1);

		read_CIFAR10(trainX, testX, trainY, testY);

		cout << "Cifar10 loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;


		timer.reset();


		for (int i = 0; i < 50000; i++) {
			inputs.push_back(trainX.col(i));

			vector<float> tempvec(10);
			for (int j = 0; j < 10; j++) {
				if (j == trainY.col(i).at<float>(0))
					tempvec[j] = (float)1.0;
				else
					tempvec[j] = (float) 0.0;
			}
			targets.push_back(tempvec);
		}
		for (int i = 0; i < 10000; i++) {
			testinputs.push_back(testX.col(i));
			testtargets.push_back(testY.col(i).at<float>(0));
		}

		cout << "Cifar10 converted in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;

		timer.reset();

		// runCNN(inputs, targets, testinputs, testtargets, timer);
		runMiniVGG(inputs, targets, testinputs, testtargets, timer);

/*
		////////////////////////////////////////////////////////////////////////////////

		/// FCNN
		//////////////////////////////////////////////////////

	   ConvNN m_nn;

	   vector<int> netVec;
	   netVec = { 1024,10 };
	   m_nn.createFullyConnectedNN(netVec, 1, 32);


	   //m_nn.forwardFCNN(inputs[0]);


	   m_nn.trainFCNN(inputs, targets, testinputs, testtargets, 50000);

	   cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << endl;

	   m_nn.trainingAccuracy(testinputs, testtargets, 2000, 1);
	   ///////////////////////////////////////////////////////////////
*/
	}
	
/*
	catch (cl::Error e) 
	{
		cout << "opencl error: " << e.what() << endl;
		cout << "error number: " << e.err() << endl;
	}
*/
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	
	// 在程序結束時清理 OpenCL 資源
	OpenCL::cleanup_OpenCL();
	
	return 0;
}