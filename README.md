# 项目描述
![logo](./faQnet.png)
**faQnet** /fɑːkjuːnet/是一个c++神经网络框架，它使用OpenCV库，并且使用单列矩阵储存输入输出。"faQnet"这个名字是"flexible and Quick neural network"的缩写，旨在实现一个强调灵活性和快速开发的神经网络框架，能够适应各种神经网络架构，同时提升开发效率。  

# 安装和使用说明  
1. 安装OpenCV库，并设置环境变量。
2. 下载faQnet，解压到任意目录。
3. 在需要使用faQnet的文件中，引用头文件/src/faQnet.h	
```c++
	#include "/src/faQnet.h"  
```

4. 在编译时，将/src/faQnet.cpp加入编译。  
```shell
	g++ -o test test.cpp /src/faQnet.cpp  
```

现在，你就可以使用faQnet了！

# 文档与学习  
如果您想要学习框架的使用方法，请参考[用户文档](./docs/user_doc.md)。  
如果您想要了解框架的内部原理，或者想要参与开发，请参考[开发文档](./docs/dev_doc.md)。  

# 快速开始/示例和代码片段

以下以/demo/Breast Cancer/Breast Cancer.cpp为例，展示如何使用faQnet。  

1. 引用头文件
```c++
	#include "faQnet.h"
```

值得一提的是，您无需引用任何C++ 标准模板库(STL)头文件，因为faQnet.h已经包含了所有STL头文件。  

2. 导入数据  
	在faQnet中，数据以单列矩阵的形式导入。因此我们内置了`load_data()`函数，用于从csv文件中导入数据。  
```c++
	std::vector<cv::Mat> input =faQnet::load_data("wdbc.csv", 	4, 33);
	std::vector<cv::Mat> target = faQnet::load_data("wdbc.csv", 2, 3);
```

3.  构建网络结构  
	在faQnet中，我们使用`faQnet::net`类来构建网络结构。您只需要将储存每一层节点数和激活函数类型的vector传入构造函数即可。  
```c++
	std::vector<int> layer_size = {30, 15, 2};
	std::vector<std::string> activation_function = {"softsign", "leaky_relu","none"};
		
	faQnet::net net(layer_size, activation_function);   
```

4. 初始化矩阵  
	在faQnet中，我们使用您构建的net对象的`init_bias()`和`init_weight()`方法来初始化偏置项矩阵和权值矩阵。只需传入初始化方法和对应参数即可。  
```c++
	net.init_bias("uniform", -0.1, 0.1);
	net.init_weight("normal", 0, 0.5);
```

5. （可选）数据归一化预处理  
	在faQnet中，我们使用net对象的`preprocess_input()`方法对输入数据进行归一化预处理。  
```c++
	net.preprocess_input(input);
```

6. 训练网络  
	在faQnet中，我们使用net对象的`train()`方法对网络进行训练。只需传入输入数据、预期输出、学习率、训练次数、采用的损失函数即可。  
```c++
	for (int i = 0; i < input.size()-100; i++){
		std::cout << "训练数据：" << i+1 <<"/" << input.size() << std::endl;
		net.train(input[i], target[i], 0.0001 ,10,"ce");
	}
```

7. 预测
	在faQnet中，我们使用net对象的`predict()`方法对数据进行预测。只需传入输入数据即可。同时，您还可以使用`faQnet::softmax()`函数对输出进行softmax处理。  
```c++
	for (int i = input.size()-100; i < input.size(); i++){
		std::cout << "预测数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << faQnet::softmax(net.predict(input[i])) << std::endl;
		std::cout << "实际数据：" << i-input.size()+101 <<"/" << 100 ;
		std::cout << target[i] << std::endl;
	}
```

# 项目结构和文件组织

|文件/目录	| 描述	| 用途 |
|-------|--------|------|
|/src	|源代码目录	|存放项目的源代码|
|/docs	|文档目录	|包含项目的文档和使用手册|
|/demo	|示例目录	|存放项目示例|
|/pics  |图片目录	|存放项目图片|
|README.md	|项目说明文件	|提供项目的基本信息和使用指南|

# 联系信息
如果你有任何问题或建议，请随时通过我的电子邮件<fQwQf6@outlook.com>或<supertjz123@foxmail.com>与我联系。