#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include "function.cpp"

using std::cout;
using std::endl;
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
		std::string act_func;


		//2024/10/9 fQwQf
		//结果矩阵
		//一个Mat对象，储存该层所有节点的前向传播中的线性运算结果。
		//规格为该层节点数*1。在反向传播中会用到。
		cv::Mat result;

		public:

		//2024/10/9 fQwQf
		//误差矩阵
		//在反向传播中计算，并用于优化。规格为该层节点数*1。
		cv::Mat error;


		//2024/10/9 fQwQf
		//构造函数：
		//传入该层节点数和下一层节点数，将成员变量的矩阵规格按上文设定。
		layer(int this_layer_node_num, int next_layer_node_num, std::string act_function = "sigmoid"){

			weight = cv::Mat::zeros(next_layer_node_num, this_layer_node_num, CV_32FC1);
			bias = cv::Mat::zeros(next_layer_node_num, 1, CV_32FC1);
			result = cv::Mat::zeros(next_layer_node_num, 1, CV_32FC1);
			error = cv::Mat::zeros(this_layer_node_num, 1, CV_32FC1);
			act_func = act_function;

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

		//2024/10/12 fQwQf
		//单层初始化权值矩阵
		//这个函数接受一个初始化方法，将权值矩阵进行初始化。
		void init_weight(std::string init_method,float a,float b=0){
			if(init_method == "uniform"){
				uniform_init(a, b, weight);
			}
			else if(init_method == "normal"){
				normal_init(a, b, weight);
			}
			else if(init_method == "constant"){
				constant_init(a, weight);
			}
		}

		//2024/10/12 fQwQf
		//单层初始化偏置矩阵
		//这个函数接受一个初始化方法，将权值矩阵进行初始化。
		void init_bias(std::string init_method,float a,float b=0){
			if(init_method == "uniform"){
				uniform_init(a, b, bias);
			}
			else if(init_method == "normal"){
				normal_init(a, b, bias);
			}
			else if(init_method == "constant"){
				constant_init(a, bias);
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
			result = weight * input + bias;
			return activation_function(result);
		}



		



		//2024/10/10 fQwQf
		/*单层反向传播
		这个函数接受上一层的误差矩阵，
		首先将其与这一层的激活函数在结果矩阵处的导函数值进行点乘得到一个中间矩阵，
		然后将其与该层权值矩阵的转置矩阵相乘即得到该层的误差矩阵，将其输出并保存。
		传入参数：
		上一层的误差矩阵*/
		cv::Mat backward(cv::Mat last_error){
			cv::Mat temp = last_error.mul(activation_function_derivative(result));
			error = weight.t() * temp;
			return error;
		}


		//2024/10/10 fQwQf
		/*单层权值更新
		这个函数接受一个学习率和上层误差矩阵。
		权值更新的具体操作是：
		将权值矩阵减去（学习率*上层误差矩阵*该层输出矩阵的转置矩阵mul这一层的激活函数在结果矩阵处的导函数值）
		。注：只需将结果矩阵代入激活函数即可得到输出矩阵。
		传入参数：
		学习率
		这是一个double型变量，代表学习率。
		上层误差矩阵*/
		void update_weight(double learning_rate, cv::Mat last_error){//last_error其实是本层误差矩阵
			cv::Mat output_derivative = activation_function_derivative(last_error, act_func);
			cv::Mat delta = last_error.mul(output_derivative);
			delta =  result * delta.t() ;
			weight -= learning_rate * delta;
		}


		//2024/10/10 fQwQf
		/*单层偏置更新
		这个函数接受一个学习率和上层误差矩阵。
		偏置项更新的具体操作是：将偏置项矩阵减去（学习率*上层误差矩阵）。
		传入参数：
		学习率
		这是一个double型变量，代表学习率。
		上层误差矩阵*/
		void update_bias(double learning_rate, cv::Mat last_error){
			bias -= learning_rate * last_error;
		}


		//2024/10/12 fQwQf
		//输出本层详情
		//今天测试中遇到了一些问题，因此我设计了这个函数，以便调试。
		void print(){
			std::cout << "weight:" << std::endl << weight << std::endl;
			std::cout << "bias:" << std::endl << bias << std::endl;
			std::cout << "result:" << std::endl << result << std::endl;
			std::cout << "error:" << std::endl << error << std::endl;
		}


		
		
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

		//构造函数
		//这个函数接受每一层每一层节点详情，然后借此初始化神经网络。
		//这里所谓初始化神经网络的含义是：在成员变量层生成每一层。
		//实际上这非常简单——只需要将该层节点数和下一层节点数及激活函数类型传入构造函数即可。
		//要注意的是最后一层没有下一层，因此不进行神经网络运算，而是直接输出或执行过滤函数（这点还没有实现）。
		//因此，最后一层不存在layers里，而是存在另一个变量里（现在还未实现）。
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

			input_mean = cv::Mat::zeros(node_num[0], 1, CV_32F);
			input_std = cv::Mat::zeros(node_num[0], 1, CV_32F);
			target_mean = cv::Mat::zeros(node_num[node_num.size() - 1], 1, CV_32F);
			target_std = cv::Mat::zeros(node_num[node_num.size() - 1], 1, CV_32F);
		}


		//2024/10/10 fQwQf
		/*前向传播
		这个函数接受一个输入矩阵，然后通过调用每一层的前向传播函数，将输入矩阵传入第一层，将第一层的输出矩阵传入第二层，将第二层的输出矩阵传入第三层，以此类推，
		直到最后一层，将最后一层的输出矩阵作为整个神经网络的输出矩阵。
		传入参数：
		输入矩阵
		这是一个Mat对象，储存输入矩阵。规格为输入数据数*1。*/
		cv::Mat forward(cv::Mat input){
			for(int i = 0; i < layers.size(); i++){
				input = layers[i].forward(input);
			}
			return input;
		}


		//2024/10/10 fQwQf
		/*反向传播
		这个函数接受前向传播的输出矩阵和目标矩阵，
		首先通过输出矩阵减去目标矩阵得到最后一层的误差矩阵，
		然后通过调用每一层的反向传播函数，将误差矩阵传入倒数第二层，将倒数第二层的误差矩阵传入倒数第三层，以此类推，直到第一层。
		传入参数：
		输出矩阵
		这是一个Mat对象，即前向传播的输出。
		目标矩阵
		这是一个Mat对象，即目标输出矩阵。*/
		void backward(cv::Mat output, cv::Mat target){
			cv::Mat error = output - target;
			for(int i = layers.size() - 1; i >= 0; i--){
				error = layers[i].backward(error);
			}
		}


		//2024/10/10 fQwQf
		/*权值更新 这个函数接受一个学习率，然后循环调用每一层的权值更新函数。
		传入参数：
		学习率
		这是一个double型变量，代表学习率。*/
		void update_weight(double learning_rate){
			for(int i = 0; i < layers.size(); i++){
				std::cout << "更新第" << i << "层weight" << std::endl;
				layers[i].update_weight(learning_rate, layers[i].error);
			}
		}

 
		//2024/10/10 fQwQf
		/*偏置更新 这个函数接受一个学习率，然后循环调用每一层的偏置更新函数。
		传入参数：
		学习率
		这是一个double型变量，代表学习率。*/
		void update_bias(double learning_rate,cv::Mat error){
			for(int i = 0; i < layers.size(); i++){
				std::cout << "更新第" << i << "层bias" << std::endl;
				
				if(i == layers.size() - 1){
					layers[i].update_bias(learning_rate, error);
				}else{
				    layers[i].update_bias(learning_rate, layers[i+1].error);
				}
				
			}
		}

		//2024/10/10 fQwQf
		/*损失函数 这个函数接受前向传播的输出矩阵和目标矩阵。计算损失值主要是为了显示出来便于分析，或者是因为这样看起来比较厉害。
		实际上，损失函数直接调用在function.cpp中定义的即可。
		总感觉这样有一点………奇怪，但是这样比较方便，而且比较简单。
		传入参数：
		输出矩阵
		这是一个Mat对象，即前向传播的输出。
		目标矩阵
		这是一个Mat对象，即目标输出矩阵。*/
		float loss(cv::Mat output, cv::Mat target, std::string loss_function_name){
			return loss_function(target, output,loss_function_name);
		}

		//2024/10/10 fQwQf
		/*训练
		大的要来了！
		这个函数接受一个训练集，一个学习率，一个训练次数。
		每次训练，先将训练集传入前向传播函数为每一层生成结果矩阵，然后调用损失函数计算loss值，接着调用反向传播函数为每一层生成误差矩阵，最后完成权值更新和偏置更新。
		这个例子里采用的是固定循环次数的训练方法。另外，也可以采用当loss值小于某个值时停止训练的方法。
		传入参数：
		输入矩阵
		这是一个Mat对象，储存训练集。是一个单列矩阵，第一层节点数应当等于它的行数。
		目标矩阵
		这是一个Mat对象，储存目标输出矩阵。是一个单列矩阵，最后一层节点数应当等于它的行数。
		学习率
		这是一个double型变量，代表学习率。
		训练次数
		这是一个整数，代表训练次数。*/
		void train(cv::Mat input, cv::Mat target, double learning_rate, int train_times){
			for(int i = 0; i < train_times; i++){
				cout <<"训练次数：" << i+1 <<"/" << train_times ;//<< "  loss值: " << loss_value << endl;
				cv::Mat output = forward(input);
				//float loss_value = loss(output, target, "mse");
				backward(output, target);
				
				update_weight(learning_rate);
				update_bias(learning_rate,output-target);
				layers[0].print();
				layers[1].print();
			}
		}

		//2024/10/10 fQwQf
		/*预测
		这个函数接受一个输入矩阵，然后传入前向传播函数，将输出传入过滤函数，输出即是整个神经网络输出。
		传入参数：
		输入矩阵
		这是一个Mat对象，储存输入矩阵。是一个行数等于第一层节点数的单列矩阵。*/
		cv::Mat predict(cv::Mat input){
			cv::Mat output = forward(input);
			return output;//我们尚未实现过滤函数。
		}

		//2024/10/10 fQwQf
		/*保存模型
		这个函数接受一个文件名，然后将整个神经网络保存到该文件中。把除了结果矩阵和误差矩阵之外的变量全存下来就可以了。
		传入参数：
		文件名
		这是一个字符串，代表文件名。*/
		/*void save_model(std::string file_name){
			cv::FileStorage model(file_name, cv::FileStorage::WRITE);
			model << "layers" << layers;
			model.release();
		}*/


		//2024/10/10 fQwQf
		/*加载模型
		这个函数接受一个文件名，然后将该文件中的神经网络加载到当前对象中。把除了结果矩阵和误差矩阵之外的变量全加载进来就可以了。
		传入参数：
		文件名
		这是一个字符串，代表文件名。*/
		/*void load_model(std::string file_name){
		    cv::FileStorage fs;
			fs.open(file_name, cv::FileStorage::READ);
			fs["layers"] >> layers;
			fs.release();
		}*/
	

		//2024/10/14 fQwQf
		/*输入数据归一化预处理
		这个函数接受若干输入矩阵，然后计算每一项数据的均值和标准差。
		传入参数：
		输入矩阵（若干） 
		这是一个储存Mat对象的vector，储存若干输入矩阵。*/
		void preprocess_input(std::vector<cv::Mat> input){
			
			for(int k = 0; k < input[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < input.size(); i++){
					sum += input[i].at<float>(k, 1);
				}
				input_mean.at<float>(k,1)=sum / input.size();
			}
			
			for(int k = 0; k < input[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < input.size(); i++){
					sum += pow(input[i].at<float>(k, 1) - input_mean.at<float>(k, 1), 2);
				}
				input_std.at<float>(k,1) = sqrt(sum / input.size());
			}
		}
	

		//2024/10/14 fQwQf
		/*输出数据归一化预处理
		这个函数接受若干输出矩阵，然后计算每一项数据的均值和标准差。
		传入参数：
		输出矩阵（若干） 
		这是一个储存Mat对象的vector，储存若干输出矩阵。*/
		void preprocess_target(std::vector<cv::Mat> target){
			for(int k = 0; k < target[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < target.size(); i++){
					sum += target[i].at<float>(k, 1);
				}
				target_mean.at<float>(k,1)=sum / target.size();
			}

			for(int k = 0; k < target[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < target.size(); i++){
					sum += pow(target[i].at<float>(k, 1) - target_mean.at<float>(k, 1), 2);
				}
				target_std.at<float>(k,1) = sqrt(sum / target.size());
			}
		}
		
	
	
	
	
	};


	//2024/10/11 fQwQf
	/*数据导入  
	一般而言，现有的数据集都是存储在文件中的，因此需要将文件中的数据转化为矩阵格式导入到程序中。
	需要注意的是，导入数据不一定指明了输入和标签，且c++函数一次只能返回一个值，
	因此这个函数实际上是返回每一行指定的两个位置间的所有数据构成的一维矩阵构成的vector,应当设计相应的能指明开始读取位置和结束读取位置的变量。
	这也意味着，输入和标签要分别读取。
	注：此处输入的位置序号从1开始。*/
	std::vector<cv::Mat> load_data(std::string file_name, int start, int end){
		std::ifstream csv_data(file_name, std::ios::in);
		std::vector<cv::Mat> output;
		
		std::string line;

		//读取的第一行可能是数据，但也可能是标题，在数据量充足的情况下，舍弃一行无关紧要。
		getline(csv_data, line);
				
		
		while (getline(csv_data, line))
		{
			cv::Mat temp(end-start+1, 1, CV_32F);
			std::istringstream sin;
			std::string data;
			sin.str(line);
			for (int i = 1 ;getline(sin, data, ',')  && i <= end; i++){
				if (i >= start){
					temp.at<float>(i-start,0) = std::stof(data);
			    }
			}

			output.push_back(temp);
		}
		csv_data.close();
		
		return output;
	}




}


int main(){
	std::vector<cv::Mat> input = faQnet::load_data("winequality-white.csv", 1, 11);
	std::cout << input[0] << std::endl;
	std::vector<cv::Mat> target = faQnet::load_data("winequality-white.csv", 12, 12);
	std::cout << target[0] << std::endl;
	std::vector<int> layer_size = {11, 5, 1};
	std::vector<std::string> activation_function = {"sigmoid", "relu"};
    faQnet::net net(layer_size, activation_function);


	net.layers[0].init_bias("normal", 0, 0.1);
	net.layers[1].init_bias("normal", 0, 0.1);
	net.layers[0].init_weight("normal", 0, 1);
	net.layers[1].init_weight("normal", 0, 1);

	net.layers[0].print();
	net.layers[1].print();


	for (int i = 0; i < 100; i++){
		std::cout << "训练数据：" << i+1 <<"/" << input.size() << std::endl;
		net.train(input[i], target[i], 0.01, 10);
	}

	for (int i = 0; i < 100; i++){
		std::cout << "预测数据：" << i+1 <<"/" << 100 ;
		std::cout << net.predict(input[i]) << std::endl;
		std::cout << "实际数据：" << i+1 <<"/" << 100 ;
		std::cout << target[i] << std::endl;
	}
}