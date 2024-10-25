# 项目描述
![logo](./faQnet.png)
**faQnet** /fɑːkjuːnet/是一个c++神经网络框架，它使用OpenCV库，并且使用单列矩阵储存输入输出。"faQnet"这个名字是"flexible and Quick neural network"的缩写，旨在实现一个强调灵活性和快速开发的神经网络框架，能够适应各种神经网络架构，同时提升开发效率。  

# 安装和使用说明  
1. 安装OpenCV库，并设置环境变量。
2. 下载faQnet，解压到任意目录。
3. 在需要使用faQnet的文件中，引用头文件/src/faQnet.h	

		#include "/src/faQnet.h"  

4. 在编译时，将/src/faQnet.cpp加入编译。  

		g++ -o test test.cpp /src/faQnet.cpp  

现在，你就可以使用faQnet了！

# 快速开始/示例和代码片段

以下以/demo/Breast Cancer/Breast Cancer.cpp为例，展示如何使用faQnet。  

1. 引用头文件

		#include "faQnet.h"
	值得一提的是，您无需引用任何C++ 标准模板库(STL)头文件，因为faQnet.h已经包含了所有STL头文件。  

2. 导入数据  
	在faQnet中，数据以单列矩阵的形式导入。因此我们内置了`load_data`函数，用于从csv文件中导入数据。  

		std::vector<cv::Mat> input = faQnet::load_data("wdbc.csv", 	4, 33);
		
		std::vector<cv::Mat> target = faQnet::load_data("wdbc.csv", 2, 3);

3.  构建网络结构  
	在faQnet中，我们使用`faQnet::net`类来构建网络结构。您只需要将储存每一层节点数和激活函数类型的vector传入构造函数即可。  

		std::vector<int> layer_size = {30, 15, 2};
		
		std::vector<std::string> activation_function = {"softsign", "leaky_relu","none"};
		
		faQnet::net net(layer_size, activation_function);   

4. 初始化矩阵  
	在faQnet中，我们使用您构建的net对象的`init_bias`和`init_weight`方法来初始化偏置项矩阵和权值矩阵。只需传入初始化方法和对应参数即可。  

