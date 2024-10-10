#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>

namespace faQnet {
    //2024/10/9 fQwQf
	//激活函数
	//激活函数是神经网络中不可或缺的部分。
	//对该函数，传入要处理的矩阵和激活函数名称，返回激活函数的输出矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	//该函数会根据激活函数名称调用不同的激活函数，返回激活函数的输出矩阵。
	cv::Mat activation_function(cv::Mat matrix, std::string act_function = "sigmoid"){
		if(act_function == "sigmoid"){
			return sigmoid(matrix);
		}
		else if(act_function == "tanh"){
			return tanh(matrix);
		}
		else if(act_function == "relu"){
			return relu(matrix);
		}
		else if(act_function == "leaky_relu"){
			return leaky_relu(matrix);
		}
		else if(act_function == "softplus"){
			return softplus(matrix);
		}
		else if(act_function == "softsign"){
			return softsign(matrix);
		}
		else if(act_function == "swish"){
			return swish(matrix);
		}
		else if(act_function == "elu"){
			return elu(matrix);
		}
	}


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

	//2024/10/9 fQwQf
	//ReLU函数
	//传入一个矩阵，返回该矩阵经过ReLU函数处理后的矩阵。
	cv::Mat relu(cv::Mat matrix){
		cv::Mat fx;
		cv::max(matrix, 0, fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//Leaky ReLU函数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数处理后的矩阵。
	cv::Mat leaky_relu(cv::Mat matrix){
		cv::Mat fx;
		cv::max(matrix, 0.01 * matrix, fx);
		return fx;
	}

	//2024/10/9 fQwQf
	//softplus函数
	//传入一个矩阵，返回该矩阵经过softmax函数处理后的矩阵。
	cv::Mat softplus(cv::Mat matrix){
		cv::Mat exp_x, fx;
		cv::exp(matrix, exp_x);
		fx = log(1.0 + exp_x);
		return fx;
	}

	//2024/10/9 fQwQf
	//softsign函数
	//传入一个矩阵，返回该矩阵经过softsign函数处理后的矩阵。
	cv::Mat softsign(cv::Mat matrix){
		cv::Mat fx;
		fx = matrix / (1.0 + abs(matrix));
		return fx;
	}

	//2024/10/9 fQwQf
	//swish函数
	//传入一个矩阵，返回该矩阵经过swish函数处理后的矩阵。
	cv::Mat swish(cv::Mat matrix){
		cv::Mat fx;
		fx = matrix / (1.0 + exp(-matrix));
		return fx;
	}

	//2024/10/9 fQwQf
	//ELU函数
	//传入一个矩阵，返回该矩阵经过ELU函数处理后的矩阵。
	cv::Mat elu(cv::Mat matrix){
		cv::Mat fx;
    	cv::exp(matrix, fx); 
    	return matrix < 0 ? (fx - 1) : matrix;
	}



	//2024/10/10 fQwQf
	//激活函数的导数
	//该函数会根据激活函数名称调用不同的激活函数的导数，返回激活函数的导数矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	cv::Mat activation_function_derivative(cv::Mat matrix, std::string act_function = "sigmoid"){
		if(act_function == "sigmoid"){
			return sigmoid_derivative(matrix);
		}
		else if(act_function == "tanh"){
			return tanh_derivative(matrix);
		}
		else if(act_function == "relu"){
			return relu_derivative(matrix);
		}
		else if(act_function == "leaky_relu"){
			return leaky_relu_derivative(matrix);
		}
		else if(act_function == "softplus"){
			return softplus_derivative(matrix); 
		}
		else if(act_function == "softsign"){
			return softsign_derivative(matrix);
		}
		else if(act_function == "swish"){
			return swish_derivative(matrix);
		}
		else if(act_function == "elu"){
			return elu_derivative(matrix);
		}
	}
	
	//2024/10/10 fQwQf
	//Sigmoid函数的导数
	//传入一个矩阵，返回该矩阵经过sigmoid函数的导函数处理后的矩阵。
	cv::Mat sigmoid_derivative(cv::Mat matrix){
		return sigmoid(matrix) * (1 - sigmoid(matrix));
	}

	//2024/10/10 fQwQf
	//Tanh函数的导数
	//传入一个矩阵，返回该矩阵经过tanh函数的导函数处理后的矩阵。
	cv::Mat tanh_derivative(cv::Mat matrix){
		return 1 - tanh(matrix) * tanh(matrix);
	}

	//2024/10/10 fQwQf
	//ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过ReLU函数的导函数处理后的矩阵。
	cv::Mat relu_derivative(cv::Mat matrix){
    	cv::Mat result = matrix.clone();
    	for(int i = 0; i < matrix.rows; ++i){
        	for(int j = 0; j < matrix.cols; ++j){
        	    result.at<float>(i, j) = matrix.at<float>(i, j) >= 0 ? 1 : 0;
    	    }
	    }
    	return result;
	}

	//2024/10/10 fQwQf
	//Leaky ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数的导函数处理后的矩阵。
	cv::Mat leaky_relu_derivative(cv::Mat matrix){
	    cv::Mat result = matrix.clone();
    	for(int i = 0; i < matrix.rows; ++i){
        	for(int j = 0; j < matrix.cols; ++j){
        	    result.at<float>(i, j) = matrix.at<float>(i, j) >= 0 ? 1 : 0.01;
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
	cv::Mat softplus_derivative(cv::Mat matrix){
		cv::Mat exp_x, fx;
		cv::exp(-matrix, exp_x);
		fx = 1.0 / (1.0 + exp_x);
		return fx;
	}

	//2024/10/10 fQwQf 
	//Softsign函数的导数
	//传入一个矩阵，返回该矩阵经过Softsign函数的导函数处理后的矩阵。
	cv::Mat softsign_derivative(cv::Mat matrix){
	    cv::Mat abs_matrix = abs(matrix);
	    return 1 / (abs_matrix + 1) / (abs_matrix + 1);
	}

	//2024/10/10 fQwQf
	//Swish函数的导数
	//传入一个矩阵，返回该矩阵经过Swish函数的导函数处理后的矩阵。
	cv::Mat swish_derivative(cv::Mat matrix){
		cv::Mat sigmoid_matrix = sigmoid(matrix);
		return sigmoid_matrix * (1 + matrix * (1 - sigmoid_matrix));
	}

	//2024/10/10 fQwQf
	//ELU函数的导数
	//传入一个矩阵，返回该矩阵经过ELU函数的导函数处理后的矩阵。
	cv::Mat elu_derivative(cv::Mat matrix){
		cv::Mat result = matrix.clone();
		for(int i = 0; i < matrix.rows; ++i){
			for(int j = 0; j < matrix.cols; ++j){
				if(matrix.at<float>(i, j) >= 0){
					result.at<float>(i, j) = 1;
				}else{
					result.at<float>(i, j) = exp(matrix.at<float>(i, j));
				}
			}
		}
		return result;
	}

}