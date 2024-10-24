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

		layer(int this_layer_node_num, int next_layer_node_num, std::string act_function = "sigmoid");

		void uniform_init(float min, float max, cv::Mat &Matrix);

		void normal_init(float mean, float std, cv::Mat &Matrix);

		void constant_init(float constant, cv::Mat &Matrix);

		void init_weight(std::string init_method,float a,float b=0);

		void init_bias(std::string init_method,float a,float b=0);

		cv::Mat forward(cv::Mat input);

		cv::Mat backward(cv::Mat last_error);

		void update_weight(double learning_rate);

		void update_bias(double learning_rate);

		void print();

	};






	class net{

		protected:

		cv::Mat input_mean;

		cv::Mat input_std;

		cv::Mat target_mean;

		cv::Mat target_std;



		public:

		std::vector<layer> layers;

		net(std::vector<int> node_num, std::vector<std::string> act_function);

		cv::Mat forward(cv::Mat input);

		void backward(cv::Mat output, cv::Mat target, std::string loss_function_name="mse");

		void update_weight(double learning_rate);

		void update_bias(double learning_rate);

		cv::Mat loss(cv::Mat output, cv::Mat target, std::string loss_function_name);

		void train(cv::Mat input, cv::Mat target, double learning_rate, int train_times, std::string loss_function_name="mse");

		cv::Mat predict(cv::Mat input);

		void preprocess_input(std::vector<cv::Mat> input);

		void preprocess_target(std::vector<cv::Mat> target);

		cv::Mat normalize_input(cv::Mat input);

		cv::Mat normalize_target(cv::Mat target);

		cv::Mat denormalize_target(cv::Mat target);

		void print_network();

		void init_weight(std::string init_method,float a,float b=0);

		void init_bias(std::string init_method,float a,float b=0);

	};

	std::vector<cv::Mat> load_data(std::string file_name, int start, int end);

}

