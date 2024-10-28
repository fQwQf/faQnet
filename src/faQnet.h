#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "function.h"
namespace faQnet{
	class layer{
		protected:
		cv::Mat weight;
		cv::Mat bias;
		std::string act_func;
		cv::Mat input_val;
		cv::Mat result_val;
		public:
		cv::Mat error;
		layer(int this_layer_node_num, int next_layer_node_num, std::string act_function = "sigmoid"){
			weight = cv::Mat::zeros(next_layer_node_num, this_layer_node_num, CV_32FC1);
			bias = cv::Mat::zeros(next_layer_node_num, 1, CV_32FC1);
			error = cv::Mat::zeros(this_layer_node_num, 1, CV_32FC1);
			act_func = act_function;
		}
		void uniform_init(float min, float max, cv::Mat &Matrix){
			for(int i = 0; i < Matrix.rows; i++){
				for(int j = 0; j < Matrix.cols; j++){
					Matrix.at<float>(i, j) = min + (max - min) * (rand() / (RAND_MAX + 1.0));
				}
			}
		}
		void normal_init(float mean, float std, cv::Mat &Matrix){
			cv::randn(Matrix, mean, std);
		}
		void constant_init(float constant, cv::Mat &Matrix){
			for(int i = 0; i < Matrix.rows; i++){
				for(int j = 0; j < Matrix.cols; j++){
					Matrix.at<float>(i, j) = constant;
				}
			}
		}
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
		cv::Mat forward(cv::Mat input){
			input_val = input;
			result_val = weight * input + bias;
			return activation_function(result_val, act_func);
		}
		cv::Mat backward(cv::Mat last_error){
			error = last_error.mul(activation_function_derivative(result_val,act_func));
			return weight.t() * error;
		}
		void update_weight(double learning_rate){
			cv::Mat delta = error * input_val.t();
			weight -= learning_rate * delta;
		}
		void update_bias(double learning_rate){
			bias -= learning_rate * error;
		}
		void print(){
			std::cout << "weight:" << std::endl << weight << std::endl;
			std::cout << "bias:" << std::endl << bias << std::endl;
			std::cout << "result:" << std::endl << result_val << std::endl;
			std::cout << "error:" << std::endl << error << std::endl;
		}
	};
	class net{
		protected:
		cv::Mat input_mean;
		cv::Mat input_std;
		cv::Mat target_mean;
		cv::Mat target_std;
		public:
		std::vector<layer> layers;
		net(std::vector<int> node_num, std::vector<std::string> act_function){
			input_mean = cv::Mat::zeros(node_num[0], 1, CV_32F);
			input_std = cv::Mat::ones(node_num[0], 1, CV_32F);
			target_mean = cv::Mat::zeros(node_num[node_num.size() - 1], 1, CV_32F);
			target_std = cv::Mat::ones(node_num[node_num.size() - 1], 1, CV_32F);
			node_num.push_back(node_num[node_num.size()-1]);
			for(int i = 0; i < node_num.size() - 1; i++){
				layers.push_back(layer(node_num[i], node_num[i + 1], act_function[i]));
			}
		}
		cv::Mat forward(cv::Mat input){
			input = normalize_input(input);
			for(int i = 0; i < layers.size(); i++){
				input = layers[i].forward(input);
			}
			return input;
		}
		void backward(cv::Mat output, cv::Mat target, std::string loss_function_name="mse"){
			cv::Mat error = loss_function_derivative(target, output, loss_function_name);
			for(int i = layers.size() -1; i >= 0; i--){
				error = layers[i].backward(error);
			}
		}
		void update_weight(double learning_rate){
			for(int i = 0; i < layers.size(); i++){
				layers[i].update_weight(learning_rate);
			}
		}
		void update_bias(double learning_rate){
			for(int i = 0; i < layers.size(); i++){
				layers[i].update_bias(learning_rate);
			}
		}
		cv::Mat loss(cv::Mat output, cv::Mat target, std::string loss_function_name){
			target = normalize_target(target);
			return loss_function(target, output,loss_function_name);
		}
		void train(cv::Mat input, cv::Mat target, double learning_rate, int train_times, std::string loss_function_name="mse"){
			target = normalize_target(target);
			for(int i = 0; i < train_times; i++){
				cv::Mat output = forward(input);
				cv::Mat loss_value = loss(output, target, loss_function_name);
				std::cout <<"target:" << std::endl << target << std::endl;
				std::cout <<"output:" << std::endl << output << std::endl;
				std::cout <<"loss_function" << std::endl << loss_function_name << std::endl;
				std::cout <<"训练次数：" << i+1 <<"/" << train_times << std::endl << "  loss值: " << std::endl << loss_value << std::endl;
				backward(output, target, loss_function_name);
				update_weight(learning_rate);
				update_bias(learning_rate);
			}
		}
		cv::Mat predict(cv::Mat input){
			cv::Mat output = forward(input);
			output = denormalize_target(output);
			return output;
		}
		void normalize_preprocess_input(std::vector<cv::Mat> input){
			std::cout <<input.size();
			for(int k = 0; k < input[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < input.size(); i++){
					sum += input[i].at<float>(k, 0);
				}
				input_mean.at<float>(k,0)=sum / input.size();
			}
			for(int k = 0; k < input[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < input.size(); i++){
					sum += pow(input[i].at<float>(k, 0) - input_mean.at<float>(k, 0), 2);
				}
				input_std.at<float>(k,0) = sqrt(sum / input.size());
			}
		}
		void normalize_preprocess_target(std::vector<cv::Mat> target){
			int times = target.size();
			for(int k = 0; k < target[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < times; i++){
					sum += target[i].at<float>(k, 0);
				}
				target_mean.at<float>(k,0)=sum / times;
			}
			for(int k = 0; k < target[0].rows; k++){
				float sum = 0;
				for(int i = 0; i < times; i++){
					sum += pow(target[i].at<float>(k, 0) - target_mean.at<float>(k, 0), 2);
				}
				target_std.at<float>(k,0) = sqrt(sum / times);
			}
		}
		cv::Mat normalize_input(cv::Mat input){
			return (input - input_mean) / input_std;
		}
		cv::Mat normalize_target(cv::Mat target){
			return (target - target_mean) / target_std;
		}
		cv::Mat denormalize_target(cv::Mat target){
			return target.mul(target_std) + target_mean;
		}
		void print_net(){
			std::cout << "层数: " << layers.size() << std::endl;
			for (int i = 0; i < layers.size(); i++){
				std::cout << "第" << i+1 << "层: " << std::endl;
				layers[i].print();
			}
			std::cout << "input_mean: " << input_mean << std::endl;
			std::cout << "input_std: " << input_std << std::endl;
			std::cout << "target_mean: " << target_mean << std::endl;
			std::cout << "target_std: " << target_std << std::endl;
		}
		void init_weight(std::string init_method,float a,float b=0){
			for (int i = 0; i < layers.size(); i++){
				layers[i].init_weight(init_method, a, b);
				std::cout << "第" << i+1 << "层权值矩阵初始化完成" << std::endl;
			}
		}
		void init_bias(std::string init_method,float a,float b=0){
			for (int i = 0; i < layers.size(); i++){
				layers[i].init_bias(init_method, a, b);
			}
		}
	};
	std::vector<cv::Mat> load_data(std::string file_name, int start, int end);
}