#include "../../src/faQnet.h"

int main(){
	std::vector<cv::Mat> input = faQnet::load_data("iris.csv",1, 4);
	std::vector<cv::Mat> target = faQnet::load_data("iris.csv", 5, 7);
	std::cout << "数据导入完成" << std::endl;
	std::vector<int> layer_size = {4 , 4, 3};
	std::vector<std::string> activation_function = {"leaky_relu","elu", "sigmoid"};
	faQnet::net net(layer_size, activation_function);
	std::cout << "网络初始化完成" << std::endl;



	net.init_bias("uniform", -0.01, 0.01);
	net.init_weight("normal", 0, 0.1);

	std::cout <<input.size() << std::endl;
	net.normalize_preprocess_input(input);
	std::cout << "输入数据预处理完成" << std::endl;
	net.normalize_preprocess_target(target);
	std::cout << "输出数据预处理完成" << std::endl;

	net.print_net();

	for (int i = 0; i < input.size()-10; i++){
		std::cout << "训练数据：" << i+1 <<"/" << input.size()-10 << std::endl;
		net.train(input[i], target[i], 0.0005 ,1000,"mse");
	}

	net.print_net();

	for (int i = input.size()-10; i < input.size(); i++){
		std::cout << "预测数据：" << i-input.size()+11 <<"/" << 10 ;
		std::cout << net.predict(input[i]) << std::endl;
		std::cout << "实际数据：" << i-input.size()+11 <<"/" << 10 ;
		std::cout << target[i] << std::endl;
	}
}
