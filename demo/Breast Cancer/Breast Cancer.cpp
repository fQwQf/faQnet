#include "../../src/faQnet.h"

int main(){
	std::vector<cv::Mat> input = faQnet::load_data("wdbc.csv", 	4, 33);
	std::vector<cv::Mat> target = faQnet::load_data("wdbc.csv", 2, 3);
	std::cout << "数据导入完成" << std::endl;
	std::vector<int> layer_size = {30, 15, 2};
	std::vector<std::string> activation_function = {"softsign", "leaky_relu","none"};
	faQnet::net net(layer_size, activation_function);
	std::cout << "网络初始化完成" << std::endl;

	

	net.init_bias("uniform", -0.1, 0.1);
	net.init_weight("normal", 0, 0.5);
	std::cout << "矩阵初始化完成" << std::endl;

	std::cout <<input.size() << std::endl;
	net.normalize_preprocess_input(input);
	std::cout << "输入数据预处理完成" << std::endl;
	//net.normalize_preprocess_target(target); 
	//std::cout << "输出数据预处理完成" << std::endl;

	net.print_net();

	for (int i = 0; i < input.size()-100; i++){
		std::cout << "训练数据：" << i+1 <<"/" << input.size() << std::endl;
		net.train(input[i], target[i], 0.0001 ,10,"ce");
	}

	net.print_net();

	for (int i = input.size()-100; i < input.size(); i++){
		std::cout << "预测数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << faQnet::softmax(net.predict(input[i])) << std::endl;
		std::cout << "实际数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << target[i] << std::endl;
	}
} 