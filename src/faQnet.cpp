#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>
#include<opencv2\highgui\highgui.hpp>


namespace faQnet{
    

	//2024/10/9 fQwQf
	//net类 代表整个神经网络
	//这是整个神经网络最大的类，代表整个神经网络。
	class net{
	    
		protected:

		//层
		//这是一个vector，储存layer类对象。
		//所有的层都储存在这里。
		std::vector<layer> layers;


		public:

		//构造函数
		//这个函数接受每一层每一层节点详情，然后借此初始化神经网络。
		//这里所谓初始化神经网络的含义是：在成员变量层生成每一层。
		//实际上这非常简单——只需要将该层节点数和下一层节点数及激活函数类型传入构造函数即可。
		//要注意的是最后一层没有下一层，因此不进行神经网络运算，而是直接输出或执行过滤函数（这点还没有实现）。  
		//传入参数：
		//传入参数：
		//每一层节点数
		//这是一个vector，储存一些整数。
		//每一层激活函数类型
		//这是一个vector，储存一些字符串。
		net(std::vector<int> node_num, std::vector<std::string> act_function){

			for(int i = 0; i < node_num.size() - 1; i++){
				layers.push_back(layer(node_num[i], node_num[i + 1], act_function[i]));
			}
		}


	}





	//2024/10/9 fQwQf
	//layer类 代表神经网络的一层
	//这是神经网络中每一层的类，代表神经网络中的一层。
	//每一层都有权重，偏置，激活函数等属性。
	//每一层都有前向传播和反向传播的方法。
	//尽管神经网络是以神经元节点为基本单位，但是只要有一定的线性代数基础，就不难注意到：
	//可以通过将一层的所有节点的数据存入一个矩阵，将基本单位从节点提升到层，以达到简化的目的
	//同时还可以调用已有的库中对于矩阵运算的方法，实现性能优化。
	//这里我们采用openCV的Mat对象实现矩阵。
	//（也可以用UMat对象，这样可以在计算机配置了openCL的情况下利用GPU加速，但是我觉得没必要）
	class layer{
		

		protected:

		//2024/10/9 fQwQf
		//权值矩阵
		//一个Mat对象，储存该层所有节点对下一层所有节点的权值。
		//规格为该层节点数*下一层节点数 。（这样处理的意义见下文。）
		cv::Mat weight;


		//2024/10/9 fQwQf
		//偏置矩阵
		//一个Mat对象，储存该层的偏置项。
		//规格为下一层节点数*1。
		//（实际上，每一项代表该层所有节点到下一层某一节点的该层节点偏置项之和。这样处理的意义见下文。）
		cv::Mat bias;


		//2024/10/9 fQwQf
		//激活函数
		//一个字符串，用于储存该层所用的激活函数。 
		std::string activation_function;


		//2024/10/9 fQwQf
		//结果矩阵
		//一个Mat对象，储存该层所有节点的前向传播中的线性运算结果。
		//规格为该层节点数*1。在反向传播中会用到。
		cv::Mat result;


		//2024/10/9 fQwQf
		//误差矩阵
		//在反向传播中计算，并用于优化。规格为该层节点数*1。
		cv::Mat error;

		public:

		//2024/10/9 fQwQf
		//构造函数：
		//传入该层节点数和下一层节点数，将成员变量的矩阵规格按上文设定。
		layer(int this_layer_node_num, int next_layer_node_num, std::string act_function = "sigmoid"){

			weight = cv::Mat::zeros(this_layer_node_num, next_layer_node_num, CV_32FC1);
			bias = cv::Mat::zeros(next_layer_node_num, 1, CV_32FC1);
			result = cv::Mat::zeros(this_layer_node_num, 1, CV_32FC1);
			error = cv::Mat::zeros(this_layer_node_num, 1, CV_32FC1);
			activation_function = act_function;

		}



		//2024/10/9 fQwQf
		//初始化矩阵
		//实际上这并不是一个函数，而是一类函数，因为有几种不同的方法用于初始化。按计划，我们应当实现所有常见的初始化方法，即：均匀分布初始化，正态分布初始化，常数初始化。另外，也有两个矩阵需要初始化。
		//下面是三种初始化方法：

		//均匀分布初始化
		//传入均匀分布的最小值和最大值和需要初始化的矩阵，将矩阵中的每一项赋值为一个均匀分布的随机值。
		void uniform_init(float min, float max, cv::Mat &matrix){
			for(int i = 0; i < matrix.rows; i++){
				for(int j = 0; j < matrix.cols; j++){
					matrix.at<float>(i, j) = min + (max - min) * (rand() / (RAND_MAX + 1.0));
				}
			}
		}
		//其实这个完全可以用opencv的randu函数实现，但是为了学习，我还是自己写了。
		
		//正态分布初始化
		//传入正态分布的均值和标准差和需要初始化的矩阵，将矩阵中的每一项赋值为一个正态分布的随机值。
		void normal_init(float mean, float std, cv::Mat &matrix){
			cv:randn(matrix, mean, std);
		}
		
		//摆烂了。

		//常数初始化
		//传入一个常数和需要初始化的矩阵，将矩阵中的每一项赋值为该常数。
		void constant_init(float constant, cv::Mat &matrix){
			for(int i = 0; i < matrix.rows; i++){
				for(int j = 0; j < matrix.cols; j++){
					matrix.at<float>(i, j) = constant;
				}
			}
		}



		//2024/10/9 fQwQf
		//单层前向传播
		/*这个函数接受一个输入矩阵，然后通过线型运算和激活函数运算，将激活函数的输出矩阵作为该层的输出矩阵。
		具体而言，该函数首先将输入矩阵与权值矩阵相乘，然后加上偏置项矩阵，得到一个中间矩阵，
		将中间矩阵保存为结果矩阵，然后将其代入激活函数，将激活函数的输出矩阵作为该层的输出。
		从这里我们可以发现：权值矩阵的规格为该层节点数*下一层节点数，
		这是因为权值矩阵的每一列列代表一个节点，每一行代表对下一层某一个节点的权重；
		偏置项矩阵的规格为下一层节点数*1，这是因为偏置项矩阵的每一项代表下一层某一节点的偏置项。 
		传入参数：
		输入矩阵
		这是一个Mat对象，储存输入矩阵。规格为输入数据数*1。
		输出矩阵
		这是一个Mat对象，储存输出矩阵。*/
		cv::Mat forward(cv::Mat input){
			result = input * weight + bias;
			return activation_function(result);
		}
		

	}


	//2024/10/9 fQwQf
	//激活函数
	//激活函数是神经网络中不可或缺的部分。
	//对该函数，传入要处理的矩阵和激活函数名称，返回激活函数的输出矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。


	//2024/10/9 fQwQf
	//sigmoid函数
	//传入一个矩阵，返回该矩阵经过sigmoid函数处理后的矩阵。
	cv::Mat sigmoid(cv::Mat matrix){
		cv::Mat exp_x, fx;
		cv::exp(-matrix, exp_x);
		fx = 1.0 / (1.0 + exp_x);
		return fx;
	}

	//2024/10/9 fQwQf
	//tanh函数
	//传入一个矩阵，返回该矩阵经过tanh函数处理后的矩阵。
	cv::Mat tanh(cv::Mat matrix){
		cv::Mat exp_x, exp_neg_x, fx;
		cv::exp(matrix, exp_x);
		cv::exp(-matrix, exp_neg_x);
		fx = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
		return fx;
	}

	

}


