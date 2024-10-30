#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "function.h"
namespace faQnet{

	//2024/10/9 fQwQf
	/*layer类 代表神经网络的一层
	这是神经网络中每一层的类，代表神经网络中的一层。
	每一层都有权重，偏置，激活函数等属性。
	每一层都有前向传播和反向传播的方法。
	尽管神经网络是以神经元节点为基本单位，但是只要有一定的线性代数基础，就不难注意到：
	可以通过将一层的所有节点的数据存入一个矩阵，将基本单位从节点提升到层，以达到简化的目的
	同时还可以调用已有的库中对于矩阵运算的方法，实现性能优化。
	这里我们采用openCV的Mat对象实现矩阵。
	（也可以用UMat对象，这样可以在计算机配置了openCL的情况下利用GPU加速，但是我觉得没必要）*/
	class layer{
		protected:

		//2024/10/9 fQwQf
		/*权值矩阵
		一个Mat对象，储存该层所有节点对下一层所有节点的权值。
		规格为该层节点数*下一层节点数 。*/
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
		std::string act_func;

		//2024/10/23 fQwQf
		//输入矩阵
		//一个Mat对象，储存该层所有节点的前向传播中的输入。规格为该层节点数*1。在反向传播中会用到。
		cv::Mat input_val;

		//2024/10/9 fQwQf
		//结果矩阵
		//一个Mat对象，储存该层所有节点的前向传播中的线性运算结果。
		//规格为下一层节点数*1。在反向传播中会用到。
		cv::Mat result_val;

		public:

		//2024/10/9 fQwQf
		//误差矩阵
		//储存该层线性运算的每个结果对于最终输出的损失值的偏导数。
		//在反向传播中计算。规格为下一层节点数*1。
		cv::Mat error;

		layer(int this_layer_node_num, int next_layer_node_num, std::string act_function);

		void uniform_init(float min, float max, cv::Mat &Matrix);
		void normal_init(float mean, float std, cv::Mat &Matrix);
		void constant_init(float constant, cv::Mat &Matrix);

		void init_weight(std::string init_method,float a,float b);
		void init_bias(std::string init_method,float a,float b);

		cv::Mat forward(cv::Mat input);
		cv::Mat backward(cv::Mat last_error);

		void update_weight(double learning_rate);
		void update_bias(double learning_rate);

		void print();
	};
	//2024/10/9 fQwQf
	//net类 代表整个神经网络
	//这是整个神经网络最大的类，代表整个神经网络。
	class net{

		protected:

		//输入均值矩阵
		//这是一个Mat对象，储存所有输入矩阵每一项的均值。
		cv::Mat input_mean;


		//输入标准差矩阵
		//这是一个Mat对象，储存所有输入矩阵每一项的标准差。
		cv::Mat input_std;


		//输出均值矩阵
		//这是一个Mat对象，储存所有输出矩阵每一项的均值。
		cv::Mat target_mean;


		//输出标准差矩阵
		//这是一个Mat对象，储存所有输出矩阵每一项的标准差。
		cv::Mat target_std;



		public:
		//层
		//这是一个vector，储存layer类对象。
		//所有的层都储存在这里。
		std::vector<layer> layers;

		net(std::vector<int> node_num, std::vector<std::string> act_function);

		cv::Mat forward(cv::Mat input);
		void backward(cv::Mat output, cv::Mat target, std::string loss_function_name);

		void update_weight(double learning_rate);
		void update_bias(double learning_rate);

		cv::Mat loss(cv::Mat output, cv::Mat target, std::string loss_function_name);

		void train(cv::Mat input, cv::Mat target, double learning_rate, int train_times, std::string loss_function_name);

		cv::Mat predict(cv::Mat input);

		void normalize_preprocess_input(std::vector<cv::Mat> input);
		void normalize_preprocess_target(std::vector<cv::Mat> target);

		cv::Mat normalize_input(cv::Mat input);
		cv::Mat normalize_target(cv::Mat target);
		cv::Mat denormalize_target(cv::Mat target);

		void print_net();

		void init_weight(std::string init_method,float a,float b);
		void init_bias(std::string init_method,float a,float b);
	};
	std::vector<cv::Mat> load_data(std::string file_name, int start, int end);
}