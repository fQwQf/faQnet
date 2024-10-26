# faQnet 使用文档

## 概述

faQnet是一个简洁的 C++ 神经网络框架，支持多层网络的构建与训练，提供灵活的 API 供用户调用。本文档将介绍核心接口，并包含一些使用示例。注意：本文档着重于为用户了解关键接口提供帮助，具体实现方法并未提及，如果感兴趣，请参考开发文档。

## 类与接口介绍

### 1. `net` 类

`net` 是神经网络的核心类，代表整个神经网络，负责神经网络的创建,前向传播,反向传播,损失计算,以及权重和偏置的更新。

#### 1.1 构造函数
```c++
net(std::vector<int> node_num,std::vector<std::string> act_function);
```

- **功能**：初始化一个神经网络。
- **参数**:
  - `node_num`：`vector<int>` 类型，表示每层神经元的数量。例如，`{3, 5, 1}` 表示包含 3 个输入神经元,5 个第一隐藏层神经元和 1 个输出神经元的三层网络。  
  - `act_function`：`vector<string>` 类型，表示每层激活函数的类型。例如，`{"softsign", "leaky_relu","none"}` 表示第一层使用 `softsign` 激活函数，第二层使用 `leaky_relu` 激活函数，第三层不使用激活函数。  
  支持的激活函数包括：`"sigmoid"`, `"tanh"`, `"relu"`, `"leaky_relu"`, `"softsign"`, `"softplus"`, `"swish"`, `"elu"`, `"none"`。

**示例**：
```c++
std::vector<int> layer_sizes = {3, 5, 1};
std::vector<std::string> act_functions = {"softsign", "leaky_relu","none"};
net myNet(layer_sizes, act_functions);
```

#### 1.2 数据预处理函数
```c++
void preprocess_input(std::vector<cv::Mat> input);
```

- **功能**：对输入数据和目标数据进行归一化预处理。预处理后，数据在输入神经网络时会自动归一化。  
- **参数**:
  - `input`：`std::vector<cv::Mat>` 类型，表示输入数据。

#### 1.3 初始化权值矩阵函数与初始化偏置矩阵函数
```c++
void init_weight(std::string init_method,float a,float b);
void init_bias(std::string init_method,float a,float b);
```
- **功能**：根据指定的初始化方法，对网络中的权值矩阵进行初始化。
- **参数**:
  - `init_method`：`std::string` 类型，表示初始化方法的名称。支持的初始化方法包括 `"uniform"`,`"normal"`,`"constant"`。
  - `a`：`float` 类型，表示初始化参数a。  
  对于 `"uniform"` 方法，表示均匀分布的下界；对于 `"normal"` 方法，表示正态分布的均值；对于 `"constant"` 方法，表示常数值。
  - `b`：`float` 类型，表示初始化参数b。  
  对于 `"uniform"` 方法，表示均匀分布的上界；对于 `"normal"` 方法，表示正态分布的标准差；对于 `"constant"` 方法，可忽略。
- **示例**：
```c++
myNet.init_weight("uniform", -0.1, 0.1); // 使用均匀分布初始化权值矩阵
myNet.init_weight("normal", 0, 0.5); // 使用正态分布初始化权值矩阵
myNet.init_weight("constant", 0.1); // 使用常数值初始化权值矩阵
myNet.init_bias("uniform", -0.1, 0.1); // 使用均匀分布初始化偏置矩阵
myNet.init_bias("normal", 0, 0.5); // 使用正态分布初始化偏置矩阵
myNet.init_bias("constant", 0.1); // 使用常数值初始化偏置矩阵
```

#### 1.4 训练函数
```c++
void train(cv::Mat input, cv::Mat target, double learning_rate, int train_times, std::string loss_function_name);
```
- **功能**：训练神经网络，通过多次迭代，使网络输出与目标值之间的误差最小化。
- **参数**:
  - `input`：`cv::Mat` 类型，表示输入数据。
  - `target`：`cv::Mat` 类型，表示目标值。
  - `learning_rate`：`double` 类型，即学习率。
  - `train_times`：`int` 类型，表示训练的迭代次数。
  - `loss_function_name`：`std::string` 类型，可选参数，指定损失函数类型。支持 `"mae"`,`"mse"`,`"sll"`,`"mape"`,`"msle"`,`"ce"`。默认为 `"mse"`。
- **示例**：
```c++
myNet.train(input, target, 0.01, 1000, "mse"); // 使用 MSE 损失函数训练网络
```
**注意**：在训练过程中，请确保输入数据和目标数据符合网络结构的要求，否则可能会导致训练失败。

#### 1.5 预测函数

```c++
cv::Mat predict(cv::Mat input);
```

- **功能**：使用训练好的网络对输入数据进行预测。
- **参数**:
  - `input`：`cv::Mat` 类型，表示输入数据。
- **返回**：`cv::Mat` 类型的矩阵，表示网络预测的输出。

**示例**：
```c++
cv::Mat input = (cv::Mat_<double>(3, 1) << 1.0, 2.0, 3.0);
cv::Mat output = myNet.predict(input);
std::cout << "Network output: " << output << std::endl;
```

#### 1.6 输出网络详情函数
```c++
void print_network();
```
- **功能**：输出网络结构信息，具体到层。
- **示例**：
```c++
myNet.print_network(); // 输出网络结构信息
```

#### 1.7 非常用方法
这些方法大部分为内部使用的中间方法，基本都被封装成了常用方法，但仍然可以提供一些有用的功能。

#### *1.7.1 数据归一化与反归一化函数*
```c++
cv::Mat normalize_input(cv::Mat input);
cv::Mat normalize_target(cv::Mat target);
cv::Mat denormalize_target(cv::Mat target);
```

- **功能**：对数据进行归一化与反归一化处理。
- **参数**:
  - `input`：`cv::Mat` 类型，表示输入数据。
  - `target`：`cv::Mat` 类型，表示目标数据。
- **返回**：`cv::Mat` 类型的矩阵，表示归一化或反归一化后的数据。

**示例**：
```c++
cv::Mat normalized_input = myNet.normalize_input(input);
cv::Mat normalized_target = myNet.normalize_target(target);
cv::Mat denormalized_target = myNet.denormalize_target(normalized_target);
```

#### *1.7.2 前向传播函数*

```c++
cv::Mat forward(cv::Mat input);
```

- **功能**：将输入数据通过网络逐层向前传递，计算输出结果。
- **参数**:
  - `input`：`cv::Mat` 类型的输入矩阵，表示输入数据的特征。
- **返回**：网络的输出矩阵，`cv::Mat` 类型。

**示例**：
```c++
cv::Mat input = (cv::Mat_<double>(3, 1) << 1.0, 2.0, 3.0);
cv::Mat output = myNet.forward(input);
std::cout << "Network output: " << output << std::endl;
```

#### *1.7.3 反向传播函数*

```c++
void backward(cv::Mat target, cv::Mat output, std::string loss_function_name = "mse");
```

- **功能**：计算输出与目标之间的误差，并将误差反向传播到每一层以计算梯度。
- **参数**:
  - `target`：`cv::Mat` 类型，表示目标输出的矩阵。
  - `output`：`cv::Mat` 类型，表示实际输出矩阵（通常为前向传播的结果）。
  - `loss_function_name`：`std::string` 类型，可选参数，指定损失函数类型。支持 `"mse"`（均方误差）,`"mape"`（平均绝对百分比误差）,`"msle"`（均方对数误差）,`"ce"`（二元交叉熵）。默认为 `"mse"`。

**示例**：
```c++
cv::Mat target = (cv::Mat_<double>(1, 1) << 0.5);
cv::Mat output = myNet.forward(input);
myNet.backward(target, output, "mse"); // 使用 MSE 损失函数
```

#### *1.7.4 更新权重*

```c++
void update_weight(double learning_rate);
```

- **功能**：基于反向传播得到的梯度，按照学习率对每层的权重进行更新。
- **参数**:
  - `learning_rate`：`double` 类型，控制权重更新的幅度。

**示例**：
```c++
myNet.update_weight(0.01); // 使用学习率 0.01 更新权重
```

#### *1.7.5 更新偏置*

```c++
void update_bias(double learning_rate);
```

- **功能**：基于反向传播得到的梯度，按照学习率对每层的偏置进行更新。
- **参数**:
  - `learning_rate`：`double` 类型，控制偏置更新的幅度。

**示例**：
```c++
myNet.update_bias(0.01); // 使用学习率 0.01 更新偏置
```

#### *1.7.6 损失函数*

```c++
cv::Mat loss(cv::Mat output, cv::Mat target, std::string loss_function_name);
```

- **功能**：计算输出与目标值之间的损失，用于评估网络的误差。
- **参数**:
  - `output`：`cv::Mat` 类型，表示网络的实际输出。
  - `target`：`cv::Mat` 类型，表示目标值。
  - `loss_function_name`：`std::string` 类型，指定损失函数的类型。支持的类型包括 `"mse"`,`"mape"`,`"msle"`,`"ce"`。
- **返回**：`cv::Mat` 类型的矩阵，包含损失值。

**示例**：
```c++
cv::Mat loss_value = myNet.loss(output, target, "mse");
std::cout << "Loss value: " << loss_value << std::endl;
```
		
### 2. 其他函数
#### 2.1 数据集加载函数
```c++
std::vector<cv::Mat> load_data(std::string file_name, int start, int end);
```
- **功能**：从CSV文件中加载数据，并将其转换为OpenCV矩阵。
- **参数**:
  - `file_name`：`std::string` 类型，表示CSV文件的路径。
  - `start`：`int` 类型，表示要加载的数据的起始列索引。
  - `end`：`int` 类型，表示要加载的数据的结束列索引。
- **返回**：`std::vector<cv::Mat>` 类型的vector，包含从CSV文件中加载的一系列数据。注意：csv文件的第一行将被跳过，因为有可能为表头。

**示例**：
```c++
std::vector<cv::Mat> input = faQnet::load_data("wdbc.csv", 4, 33);
std::vector<cv::Mat> target = faQnet::load_data("wdbc.csv", 2, 3);
```


