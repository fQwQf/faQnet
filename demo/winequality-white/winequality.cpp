#include "../../src/faQnet.h"

int main(){
	std::vector<cv::Mat> input = faQnet::load_data("winequality-white.csv",1, 11);
	std::vector<cv::Mat> target = faQnet::load_data("winequality-white.csv", 12, 12);
	std::cout << "数据导入完成" << std::endl;
	std::vector<int> layer_size = {11, 6 , 1};
	std::vector<std::string> activation_function = {"tanh", "relu","none"};
	faQnet::net net(layer_size, activation_function);
	std::cout << "网络初始化完成" << std::endl;

	

	net.init_bias("uniform", -0.01, 0.01);
	net.init_weight("normal", 0, 0.2);
	std::cout << "矩阵初始化完成" << std::endl;

	std::cout <<input.size() << std::endl;
	net.normalize_preprocess_input(input);
	std::cout << "输入数据预处理完成" << std::endl;
	net.normalize_preprocess_target(target);
	std::cout << "输出数据预处理完成" << std::endl;

	net.print_net();

	for (int i = 0; i < input.size()-100; i++){
		std::cout << "训练数据：" << i+1 <<"/" << input.size() << std::endl;
		net.train(input[i], target[i], 0.0005 ,100,"mse");
	}

	net.print_net();

	for (int i = input.size()-100; i < input.size(); i++){
		std::cout << "预测数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << net.predict(input[i]) << std::endl;
		std::cout << "实际数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << target[i] << std::endl;
	}
} 