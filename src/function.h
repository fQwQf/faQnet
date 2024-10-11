#include<bits/stdc++.h>
#include<opencv2/core/core.hpp>

//这个文件储存了神经网络中用到的的激活函数，激活函数的导函数，以及损失函数的声明。

namespace faQnet {
	//2024/10/9 fQwQf
	//激活函数
	//激活函数是神经网络中不可或缺的部分。
	//对该函数，传入要处理的矩阵和激活函数名称，返回激活函数的输出矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	//该函数会根据激活函数名称调用不同的激活函数，返回激活函数的输出矩阵。
	cv::Mat activation_function(cv::Mat matrix, std::string act_function = "sigmoid");

	//2024/10/9 fQwQf
	//sigmoid函数
	//传入一个矩阵，返回该矩阵经过sigmoid函数处理后的矩阵。
	cv::Mat sigmoid(cv::Mat matrix);

	//2024/10/9 fQwQf
	//tanh函数
	//传入一个矩阵，返回该矩阵经过tanh函数处理后的矩阵。
	cv::Mat tanh(cv::Mat matrix);

	//2024/10/9 fQwQf
	//ReLU函数
	//传入一个矩阵，返回该矩阵经过ReLU函数处理后的矩阵。
	cv::Mat relu(cv::Mat matrix);

	//2024/10/9 fQwQf
	//Leaky ReLU函数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数处理后的矩阵。
	cv::Mat leaky_relu(cv::Mat matrix);

	//2024/10/9 fQwQf
	//softplus函数
	//传入一个矩阵，返回该矩阵经过softmax函数处理后的矩阵。
	cv::Mat softplus(cv::Mat matrix);

	//2024/10/9 fQwQf
	//softsign函数
	//传入一个矩阵，返回该矩阵经过softsign函数处理后的矩阵。
	cv::Mat softsign(cv::Mat matrix);

	//2024/10/9 fQwQf
	//swish函数
	//传入一个矩阵，返回该矩阵经过swish函数处理后的矩阵。
	cv::Mat swish(cv::Mat matrix);

	//2024/10/9 fQwQf
	//ELU函数
	//传入一个矩阵，返回该矩阵经过ELU函数处理后的矩阵。
	cv::Mat elu(cv::Mat matrix);


	//2024/10/10 fQwQf
	//激活函数的导数
	//该函数会根据激活函数名称调用不同的激活函数的导数，返回激活函数的导数矩阵。
	//激活函数名称是一个字符串，储存激活函数名称。
	cv::Mat activation_function_derivative(cv::Mat matrix, std::string act_function = "sigmoid");

	//2024/10/10 fQwQf
	//Sigmoid函数的导数
	//传入一个矩阵，返回该矩阵经过sigmoid函数的导函数处理后的矩阵。
	cv::Mat sigmoid_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//Tanh函数的导数
	//传入一个矩阵，返回该矩阵经过tanh函数的导函数处理后的矩阵。
	cv::Mat tanh_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过ReLU函数的导函数处理后的矩阵。
	cv::Mat relu_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//Leaky ReLU函数的导数
	//传入一个矩阵，返回该矩阵经过Leaky ReLU函数的导函数处理后的矩阵。
	cv::Mat leaky_relu_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//Softplus函数的导数
	//传入一个矩阵，返回该矩阵经过Softplus函数的导函数处理后的矩阵。
	//聪明的你也许会发现，这个函数的导函数就是sigmoid函数。
	//那么，为什么不直接调用sigmoid函数呢？
	//这是因为，我有一个极其奇怪的代码量下限规定。
	//这是不合理的，但是，我必须这么做。
	//所以，我必须写一个一模一样的函数，来增加代码量。
	//说实话，在保证可读性的前提下，实现同样的功能，代码量应该是越少越好的。
	cv::Mat softplus_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//Softsign函数的导数
	//传入一个矩阵，返回该矩阵经过Softsign函数的导函数处理后的矩阵。
	cv::Mat softsign_derivative(cv::Mat matrix);

	//2024/10/10 fQwQf
	//Swish函数的导数
	//传入一个矩阵，返回该矩阵经过Swish函数的导函数处理后的矩阵。
	cv::Mat swish_derivative(cv::Mat matrix);
	//2024/10/10 fQwQf
	//ELU函数的导数
	//传入一个矩阵，返回该矩阵经过ELU函数的导函数处理后的矩阵。
	cv::Mat elu_derivative(cv::Mat matrix);



	//2024/10/10 fQwQf
	//损失函数
	//传入两个矩阵和采用的损失函数名称，返回两个矩阵之间的损失值。
	float loss_function(cv::Mat y_true, cv::Mat y_pred, std::string loss_function_name);


	//2024/10/10 fQwQf
	//平均绝对误差 (MAE/L1Loss)
	float mae(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//平均平方误差 (MSE/L2Loss)
	float mse(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//均方根误差 (RMSE)
	float rmse(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//平均绝对百分比误差 (MAPE)
	float mape(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//平滑平均绝对误差 (SLL/Smooth L1Loss)
	float sll(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//均方对数误差 (MSLE)
	float msle(cv::Mat y_true, cv::Mat y_pred);

	//2024/10/10 fQwQf
	//二元交叉熵损失函数（CE）
	float ce(cv::Mat y_true, cv::Mat y_pred);

}