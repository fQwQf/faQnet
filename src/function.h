#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>



//这个文件储存了神经网络中用到的各种函数（字面意义）的实现。

namespace faQnet {

	cv::Mat activation_function(cv::Mat Matrix, std::string act_function);

	cv::Mat sigmoid(cv::Mat Matrix);

	cv::Mat tanh(cv::Mat Matrix);


	cv::Mat relu(cv::Mat Matrix);


	cv::Mat leaky_relu(cv::Mat Matrix);

	cv::Mat softplus(cv::Mat Matrix);

	cv::Mat softsign(cv::Mat Matrix);

	cv::Mat swish(cv::Mat Matrix);

	cv::Mat elu(cv::Mat Matrix);

	cv::Mat softmax(cv::Mat Matrix);




	cv::Mat activation_function_derivative(cv::Mat Matrix, std::string act_function);

	cv::Mat sigmoid_derivative(cv::Mat Matrix);

	cv::Mat tanh_derivative(cv::Mat Matrix);

	cv::Mat relu_derivative(cv::Mat Matrix);

	cv::Mat leaky_relu_derivative(cv::Mat Matrix);

	cv::Mat softplus_derivative(cv::Mat Matrix);

	cv::Mat softsign_derivative(cv::Mat Matrix);

	cv::Mat swish_derivative(cv::Mat Matrix);

	cv::Mat elu_derivative(cv::Mat Matrix);

	cv::Mat softmax_derivative(cv::Mat Matrix);

	cv::Mat loss_function(cv::Mat y_true, cv::Mat y_pred, std::string loss_function_name);

	cv::Mat mse(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat mae(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat mape(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat sll(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat msle(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat ce(cv::Mat y_true, cv::Mat y_pred);



	cv::Mat loss_function_derivative(cv::Mat y_true, cv::Mat y_pred, std::string loss_function_name);

	cv::Mat mae_derivative(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat mse_derivative(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat mape_derivative(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat sll_derivative(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat msle_derivative(cv::Mat y_true, cv::Mat y_pred);

	cv::Mat ce_derivative(cv::Mat y_true, cv::Mat y_pred);



}