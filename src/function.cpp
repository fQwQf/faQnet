#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>

#include"function.h"

//这个文件储存了神经网络中用到的的激活函数，激活函数的导函数，以及损失函数的实现。

namespace faQnet {
	//2024/10/9 fQwQf
	//激活函数
	//激活函数是神经网络中不可或缺的部分。
	//对该函数，传入要处理的矩阵和激活函数名称，返回激活函数的输出矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	//该函数会根据激活函数名称调用不同的激活函数，返回激活函数的输出矩阵。
	cv::Mat activation_function(cv::Mat Matrix, std::string act_function){
		if(act_function == "sigmoid"){
			return sigmoid(Matrix);
		}
		else if(act_function == "tanh"){
			return tanh(Matrix);
		}
		else if(act_function == "relu"){
			return relu(Matrix);
		}
		else if(act_function == "leaky_relu"){
			return leaky_relu(Matrix);
		}
		
		else if(act_function == "softsign"){
			return softsign(Matrix);
		}
		else if(act_function == "softplus"){
			return softplus(Matrix);
		}
		else if(act_function == "swish"){
			return swish(Matrix);
		}
		else if(act_function == "elu"){
			return elu(Matrix);
		}
		return Matrix;
	}


	//2024/10/9 fQwQf
	//sigmoid函数
	//传入一个矩阵，返回该矩阵经过sigmoid函数处理后的矩阵。
	cv::Mat sigmoid(cv::Mat Matrix){
		cv::Mat exp_x, fx;
		cv::exp(-Matrix, exp_x);
		fx = 1.0 / (1.0 + exp_x);
		return fx;
	}

	//2024/10/9 fQwQf
	//tanh函数
	//传入一个矩阵，返回该矩阵经过tanh函数处理后的矩阵。
	cv::Mat tanh(cv::Mat Matrix){
		cv::Mat exp_x, exp_neg_x, fx;
		cv::exp(Matrix, exp_x);
		cv::exp(-Matrix, exp_neg_x);
		fx = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
		return fx;
	}

	//2024/10/9 fQwQf
	//ReLU函数
	//传入一个矩阵，返回该矩阵经过ReLU函数处理后的矩阵。
	cv::Mat relu(cv::Mat Matrix){
		cv::Mat fx;
		cv::max(Matrix, 0, fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//Leaky ReLU函数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数处理后的矩阵。
	cv::Mat leaky_relu(cv::Mat Matrix){
		cv::Mat fx;
		cv::max(Matrix, 0.01 * Matrix, fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//softplus函数
	//传入一个矩阵，返回该矩阵经过softplus函数处理后的矩阵。
	cv::Mat softplus(cv::Mat Matrix){
		cv::Mat exp_x, fx;
		cv::exp(Matrix, exp_x);
		cv::log((1.0 + exp_x),fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//softsign函数
	//传入一个矩阵，返回该矩阵经过softsign函数处理后的矩阵。
	cv::Mat softsign(cv::Mat Matrix){
		cv::Mat fx;
		fx = Matrix / (1.0 + abs(Matrix));
		return fx;
	}

	//2024/10/9 fQwQf
	//swish函数
	//传入一个矩阵，返回该矩阵经过swish函数处理后的矩阵。
	cv::Mat swish(cv::Mat Matrix){
		cv::Mat fx;
		cv::exp(-Matrix, fx);
		fx = Matrix / (1.0 + fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//ELU函数
	//传入一个矩阵，返回该矩阵经过ELU函数处理后的矩阵。
	cv::Mat elu(cv::Mat Matrix){
		cv::Mat exp_matrix;
		cv::exp(Matrix, exp_matrix);
		for (int i = 0; i < Matrix.rows; ++i){
			for(int j = 0; j < Matrix.cols; ++j){
				if(Matrix.at<float>(i, j) < 0){
					Matrix.at<float>(i, j) = exp_matrix.at<float>(i, j) - 1;
				}
			}
		}
		return Matrix;
	}




	//2024/10/10 fQwQf
	//激活函数的导数
	//该函数会根据激活函数名称调用不同的激活函数的导数，返回激活函数的导数矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	cv::Mat activation_function_derivative(cv::Mat Matrix, std::string act_function){
		if(act_function == "sigmoid"){
			return sigmoid_derivative(Matrix);
		}
		else if(act_function == "tanh"){
			return tanh_derivative(Matrix);
		}
		else if(act_function == "relu"){
			return relu_derivative(Matrix);
		}
		else if(act_function == "leaky_relu"){
			return leaky_relu_derivative(Matrix);
		}
		else if(act_function == "softsign"){
			return softsign_derivative(Matrix);
		}
		else if(act_function == "softplus"){
			return softplus_derivative(Matrix);
		}
		else if(act_function == "swish"){
			return swish_derivative(Matrix);
		}
		else if(act_function == "elu"){
			return elu_derivative(Matrix);
		}
		return Matrix;
	}

	//2024/10/10 fQwQf
	//Sigmoid函数的导数
	//传入一个矩阵，返回该矩阵经过sigmoid函数的导函数处理后的矩阵。
	cv::Mat sigmoid_derivative(cv::Mat Matrix){
		return sigmoid(Matrix).mul((1 - sigmoid(Matrix))) ;
	}

	//2024/10/10 fQwQf
	//Tanh函数的导数
	//传入一个矩阵，返回该矩阵经过tanh函数的导函数处理后的矩阵。
	cv::Mat tanh_derivative(cv::Mat Matrix){
		return 1 - tanh(Matrix) * tanh(Matrix);
	}

	//2024/10/10 fQwQf
	//ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过ReLU函数的导函数处理后的矩阵。
	cv::Mat relu_derivative(cv::Mat Matrix){
		cv::Mat result = Matrix.clone();
		for(int i = 0; i < Matrix.rows; ++i){
			for(int j = 0; j < Matrix.cols; ++j){
				result.at<float>(i, j) = Matrix.at<float>(i, j) >= 0 ? 1 : 0;
			}
		}
		return result;
	}

	//2024/10/10 fQwQf
	//Leaky ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数的导函数处理后的矩阵。
	cv::Mat leaky_relu_derivative(cv::Mat Matrix){
		cv::Mat result = Matrix.clone();
		for(int i = 0; i < Matrix.rows; ++i){
			for(int j = 0; j < Matrix.cols; ++j){
				if (Matrix.at<float>(i, j) >= 0){
					result.at<float>(i, j) = 1;
				}else{
					result.at<float>(i, j) = 0.01;
				}
			}
		}
		return result;
	}

	//2024/10/10 fQwQf
	//Softplus函数的导数
	//传入一个矩阵，返回该矩阵经过Softplus函数的导函数处理后的矩阵。
	//聪明的你也许会发现，这个函数的导函数就是sigmoid函数。
	//那么，为什么不直接调用sigmoid函数呢？
	//这是因为，我有一个极其奇怪的代码量下限规定。
	//这是不合理的，但是，我必须这么做。
	//所以，我必须写一个一模一样的函数，来增加代码量。
	//说实话，在保证可读性的前提下，实现同样的功能，代码量应该是越少越好的。
	cv::Mat softplus_derivative(cv::Mat Matrix){
		cv::Mat exp_x, fx;
		cv::exp(-Matrix, exp_x);
		fx = 1.0 / (1.0 + exp_x);
		return fx;
	}

	//2024/10/10 fQwQf
	//Softsign函数的导数
	//传入一个矩阵，返回该矩阵经过Softsign函数的导函数处理后的矩阵。
	cv::Mat softsign_derivative(cv::Mat Matrix){
		cv::Mat abs_Matrix = abs(Matrix);
		return 1 / (abs_Matrix + 1) / (abs_Matrix + 1);
	}

	//2024/10/10 fQwQf
	//Swish函数的导数
	//传入一个矩阵，返回该矩阵经过Swish函数的导函数处理后的矩阵。
	cv::Mat swish_derivative(cv::Mat Matrix){
		cv::Mat sigmoid_Matrix = sigmoid(Matrix);
		return sigmoid_Matrix * (1 + Matrix * (1 - sigmoid_Matrix));
	}

	//2024/10/10 fQwQf
	//ELU函数的导数
	//传入一个矩阵，返回该矩阵经过ELU函数的导函数处理后的矩阵。
	cv::Mat elu_derivative(cv::Mat Matrix){
		cv::Mat result = Matrix.clone();
		for(int i = 0; i < Matrix.rows; ++i){
			for(int j = 0; j < Matrix.cols; ++j){
				if(Matrix.at<float>(i, j) >= 0){
					result.at<float>(i, j) = 1;
				}else{
					result.at<float>(i, j) = exp(Matrix.at<float>(i, j));
				}
			}
		}
		return result;
	}



	//2024/10/11 fQwQf
	//损失函数
	//传入两个矩阵和采用的损失函数名称，返回两个矩阵之间的损失值。
	float loss_function(cv::Mat y_true, cv::Mat y_pred, std::string loss_function_name){
		if(loss_function_name == "mse"){
			return mse(y_true, y_pred);
		}else if(loss_function_name == "mae"){
			return mae(y_true, y_pred);
		}else if(loss_function_name == "sll"){
			return sll(y_true, y_pred);
		}else if(loss_function_name == "ce"){
			return ce(y_true, y_pred);
		}else if(loss_function_name == "rmse"){
			return rmse(y_true, y_pred);
		}else if(loss_function_name == "mape"){
			return mape(y_true, y_pred);
		}else if(loss_function_name == "msle"){
			return msle(y_true, y_pred);
		}
		return 0;
	}


	//2024/10/11 fQwQf
	//平均绝对误差 (MAE/L1Loss)
	float mae(cv::Mat y_true, cv::Mat y_pred){
		cv::Scalar sum = cv::sum(cv::abs(y_true - y_pred));
		return sum[0] / y_true.rows;
	}

	//2024/10/11 fQwQf
	//平均平方误差 (MSE/L2Loss)
	float mse(cv::Mat y_true, cv::Mat y_pred){
		cv::Scalar sum = cv::sum((y_true - y_pred).mul(y_true - y_pred));
		return  sum[0] / y_true.rows;
	}

	//2024/10/11 fQwQf
	//均方根误差 (RMSE)
	float rmse(cv::Mat y_true, cv::Mat y_pred){
		return std::sqrt(mse(y_true, y_pred));
	}

	//2024/10/11 fQwQf
	//平均绝对百分比误差 (MAPE)
	float mape(cv::Mat y_true, cv::Mat y_pred){
		cv::Scalar sum = cv::sum((y_true - y_pred) / y_true);
		return cv::abs(sum[0]) / y_true.rows;
	}

	//2024/10/11 fQwQf
	//平滑平均绝对误差 (SLL/Smooth L1Loss)
	float sll(cv::Mat y_true, cv::Mat y_pred){
		cv::Mat diff = y_true - y_pred;
		cv::Mat smooth_l1 = cv::Mat::zeros(diff.size(), diff.type());
		for (int i = 0; i < diff.rows; ++i) {
			for (int j = 0; j < diff.cols; ++j) {
				float val = abs(diff.at<float>(i, j));
				if (val < 1.0) {
					smooth_l1.at<float>(i, j) = 0.5 * val * val;
				} else {
					smooth_l1.at<float>(i, j) = val - 0.5;
				}
			}
		}
		cv::Scalar sum = cv::sum(smooth_l1);
		return sum[0] / y_true.rows;
	}

	//2024/10/11 fQwQf
	//均方对数误差 (MSLE)
	float msle(cv::Mat y_true, cv::Mat y_pred){
		// 计算自然对数
		cv::Mat log_y_true, log_y_pred;
		cv::log(y_true + 1, log_y_true);
		cv::log(y_pred + 1, log_y_pred);

		cv::Mat diff_squared = (log_y_true - log_y_pred).mul(log_y_true - log_y_pred);

		cv::Scalar sum = cv::sum(diff_squared);
		return sum[0] / y_true.rows;
	}

	//2024/10/11 fQwQf
	//二元交叉熵损失函数（CE）
	float ce(cv::Mat y_true, cv::Mat y_pred){
		cv::Mat result = cv::Mat::zeros(y_true.size(), y_true.type());
		for (int i = 0; i < result.rows; ++i) {
			for (int j = 0; j < result.cols; ++j) {
				float tru = y_true.at<float>(i, j);
				float pre = y_pred.at<float>(i, j);
				result.at<float>(i, j) = tru * log(pre) + (1 - tru) * log(1 - pre);
			}
		}
		cv::Scalar sum = cv::sum(result);
		return - sum[0] / y_true.rows;
	}

}



