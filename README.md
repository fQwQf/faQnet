![logo](./faQnet.png)
**faQnet** /fɑːkjuːnet/是一个c++神经网络框架，它使用OpenCV库，并且使用单列矩阵储存输入输出。"faQnet"这个名字是"flexible and Quick neural network"的缩写，旨在实现一个强调灵活性和快速开发的神经网络框架，能够适应各种神经网络架构，同时提升开发效率。  

该项目采用的思想方法取得了2024年诺贝尔物理学奖，充分体现了其前沿性和前瞻性。  

以下是开发中计划实现的架构与功能，以及较为具体的实现方法：

## net类  代表整个神经网络  
这是整个神经网络最大的类，代表整个神经网络。  
我们为了简化计算，用单列矩阵储存输入输出。显然，第一层节点的数量必须等于输入矩阵的行数，最后一层节点的数量必须等于输出矩阵的行数。  

成员变量：


- 层  
	这是一个vector，储存layer类对象。所有的层都储存在这里。layer类对象会在后文详细提到。  
	2024/10/9 fQwQf 已完成  


成员函数：  
- 构造函数  
	这个函数接受每一层每一层节点详情，然后借此初始化神经网络。这里所谓初始化神经网络的含义是：在成员变量层生成每一层。实际上这非常简单——只需要将该层节点数和下一层节点数及激活函数类型传入构造函数即可。  
	要注意的是最后一层没有下一层，因此不进行神经网络运算，而是直接输出或执行过滤函数（这点还没有实现）。  

	传入参数：
	- 每一层节点数  
	这是一个vector，储存一些整数。
	- 每一层激活函数类型  
	这是一个vector，储存一些字符串。
	2024/10/9 fQwQf 已完成  


- 前向传播  
	这个函数接受一个输入矩阵，然后通过调用每一层的前向传播函数，将输入矩阵传入第一层，将第一层的输出矩阵传入第二层，将第二层的输出矩阵传入第三层，以此类推，直到最后一层，将最后一层的输出矩阵作为整个神经网络的输出矩阵。  

	传入参数：
	- 输入矩阵  
	这是一个Mat对象，储存输入矩阵。规格为输入数据数*1。

- 反向传播  
	这个函数接受前向传播的输出矩阵和目标矩阵，首先通过输出矩阵减去目标矩阵得到一个中间矩阵，然后将其与最后一层的激活函数在结果处的导函数进行mul运算（openCV中的方法，即矩阵对应位相乘）得到最后一层的误差矩阵，然后通过调用每一层的反向传播函数，将误差矩阵传入倒数第二层，将倒数第二层的误差矩阵传入倒数第三层，以此类推，直到第一层。  

	传入参数：
	- 输出矩阵  
	这是一个Mat对象，即前向传播的输出。

	- 目标矩阵  
	这是一个Mat对象，即目标输出矩阵。 

- 损失函数
	这个函数接受前向传播的输出矩阵和目标矩阵。计算损失值主要是为了显示出来便于分析，或者是因为这样看起来比较厉害。实际上，有几种不同的损失函数，我们计划实现2种以上。  

	传入参数：
	- 输出矩阵  
	这是一个Mat对象，即前向传播的输出。

	- 目标矩阵  
	这是一个Mat对象，即目标输出矩阵。  

- 权值更新
	这个函数接受一个学习率，然后循环调用每一层的权值更新函数。  

	传入参数：
	- 学习率  
	这是一个double型变量，代表学习率。

- 偏置更新
	这个函数接受一个学习率，然后循环调用每一层的偏置更新函数。  

	传入参数：
	- 学习率  
	这是一个double型变量，代表学习率。


- 训练  
	这个函数接受一个训练集，一个学习率，一个训练次数。每次训练，先将训练集传入前向传播函数为每一层生成结果矩阵，然后调用损失函数计算loss值，接着调用反向传播函数为每一层生成误差矩阵，最后完成权值更新和偏置更新。  
	这个例子里采用的是固定循环次数的训练方法。另外，也可以采用当loss值小于某个值时停止训练的方法。我计划同时实现这两种方法。

	传入参数：
	- 输入矩阵  
	这是一个Mat对象，储存训练集。是一个单列矩阵，第一层节点数应当等于它的行数。  

	- 目标矩阵  
	这是一个Mat对象，储存目标输出矩阵。是一个单列矩阵，最后一层节点数应当等于它的行数。  
	
	- 学习率  
	这是一个double型变量，代表学习率。

	- 训练次数  
	这是一个整数，代表训练次数。  


- 保存  
	这个函数接受一个文件名，然后将整个神经网络保存到该文件中。把除了结果矩阵和误差矩阵之外的变量全存下来就可以了。  

	传入参数：
	- 文件名  
	这是一个字符串，代表文件名。

- 加载  
	这个函数接受一个文件名，然后将该文件中的神经网络加载到当前神经网络中。  

	传入参数：
	- 文件名  
	这是一个字符串，代表文件名。  

- 预测  
	这个函数接受一个输入矩阵，然后传入前向传播函数，将输出传入过滤函数，输出即是整个神经网络输出。  

	传入参数：
	- 输入矩阵  
	这是一个Mat对象，储存输入矩阵。是一个行数等于第一层节点数的单列矩阵。


## layer类  代表神经网络的一层  
尽管神经网络是以神经元节点为基本单位，但是只要有一定的线性代数基础，就不难注意到：可以通过将一层的所有节点的数据存入一个矩阵，将基本单位从节点提升到层，以达到简化的目的，同时还可以调用已有的库中对于矩阵运算的方法，实现性能优化。这里我们采用openCV的Mat对象实现矩阵。  
（也可以用UMat对象，这样可以在计算机配置了openCL的情况下利用GPU加速，但是我觉得没必要）  
值得注意的是这里对矩阵中的所有项都默认为0，因为有几种不同的方式为权值矩阵和偏置项矩阵赋值，因此最好将赋值过程处理为单独的函数。  
注意到这里实际上将激活函数视作了线性函数处理，这是为了简化计算。上过高中的人都知道：在很小的范围内，可以以直代曲近似相等的。而梯度下降的步长通常很小，因此这样处理是可行的。  


成员变量：   
- 权值矩阵  
	一个Mat对象，储存该层所有节点对下一层所有节点的权值。规格为该层节点数*下一层节点数 。（这样处理的意义见下文。）  
	2024/10/9 fQwQf 已完成  

- 偏置项矩阵  
	一个Mat对象，储存该层的偏置项。规格为下一层节点数*1。（实际上，每一项代表该层所有节点到下一层某一节点的该层节点偏置项之和。这样处理的意义见下文。） 
	2024/10/9 fQwQf 已完成  

- 激活函数  
	一个字符串，用于储存该层所用的激活函数。
	激活函数有多种，计划实现8种以上。  
	2024/10/9 fQwQf 已完成  


- 结果矩阵  
	一个Mat对象，储存该层所有节点的前向传播中的线性运算结果。规格为该层节点数*1。在反向传播中会用到。  
	2024/10/9 fQwQf 已完成  


- 误差矩阵  
	在反向传播中计算，并用于优化。规格为该层节点数*1。  
	2024/10/9 fQwQf 已完成  


成员函数

- 构造函数：  
	传入该层节点数，下一层节点数和激活函数类型，将成员变量的矩阵规格按上文设定。  

	传入参数：  
	-  该层节点数  
	- 下一层节点数  
	这两个是整数。
	- 激活函数类型（默认为"sigmoid"）  
	这个是字符串。  
	2024/10/9 fQwQf 已完成  


- 初始化矩阵  
	实际上这并不是一个函数，而是一类函数，因为有几种不同的方法用于初始化。按计划，我们应当实现所有常见的初始化方法，即：均匀分布初始化，正态分布初始化，常数初始化。另外，也有两个矩阵需要初始化。  
	下面是三种初始化方法简介：  
	- 均匀分布初始化  
	传入均匀分布的最小值和最大值和需要初始化的矩阵，将矩阵中的每一项赋值为一个均匀分布的随机值。
	- 正态分布初始化  
	传入正态分布的均值和标准差和需要初始化的矩阵，将矩阵中的每一项赋值为一个正态分布的随机值。
	- 常数初始化  
	传入一个常数和需要初始化的矩阵，将矩阵中的每一项赋值为该常数。  
2024/10/9 fQwQf 已完成  
 

- 单层前向传播  
	这个函数接受一个输入矩阵，然后通过线型运算和激活函数运算，将激活函数的输出矩阵作为该层的输出矩阵。  
	具体而言，该函数首先将输入矩阵与权值矩阵相乘，然后加上偏置项矩阵，得到一个中间矩阵，将中间矩阵保存为结果矩阵，然后将其代入激活函数，将激活函数的输出矩阵作为该层的输出。  
	从这里我们可以发现：权值矩阵的规格为该层节点数\*下一层节点数，这是因为权值矩阵的每一列列代表一个节点，每一行代表对下一层某一个节点的权重；偏置项矩阵的规格为下一层节点数\*1，这是因为偏置项矩阵的每一项代表下一层某一节点的偏置项。
	传入参数：  
	- 输入矩阵  
	这是一个Mat对象，储存输入矩阵。规格为输入数据数*1。

- 单层反向传播  
	这个函数接受上一层的误差矩阵，首先将其与这一层的激活函数在结果矩阵处的导函数值进行mul运算得到一个中间矩阵，然后将其与该层权值矩阵的转置矩阵相乘即得到该层的误差矩阵，将其输出并保存。  

	传入参数：  
	- 上一层的误差矩阵  

- 单层权值更新
	这个函数接受一个学习率和上层误差矩阵。权值更新的具体操作是：将权值矩阵减去（学习率\*上层误差矩阵\*该层输出矩阵的转置矩阵mul这一层的激活函数在结果矩阵处的导函数值）。注：只需将结果矩阵代入激活函数即可得到输出矩阵。  

	传入参数：  
	- 学习率  
	这是一个double型变量，代表学习率。
	- 上层误差矩阵  

- 单层偏置更新
	这个函数接受一个学习率和上层误差矩阵。偏置项更新的具体操作是：将偏置项矩阵减去（学习率\*上层误差矩阵）。  

	传入参数：  
	- 学习率  
	这是一个double型变量，代表学习率。
	- 上层误差矩阵  

## 激活函数
激活函数是神经网络中不可或缺的部分。  
激活函数有多种，计划实现8种以上。  
下面是几种常见的激活函数简介：  
- Sigmoid函数  
	这是最常见的一种激活函数。  
	公式：   
	$${ f }(x)=\sigma (x)=\frac { 1 }{ 1+{ e }^{ -x } } $$
	导函数：  
	$${ f }^{ ' }(x)=f(x)(1-f(x))$$  
- Tanh函数  
	公式：  
	$${ f }(x)=tanh(x)=\frac { { e }^{ x }-{ e }^{ -x } }{ { e }^{ x }+{ e }^{ -x } }$$
	导函数：  
	$${ f }^{ ' }(x)=1-f(x)^{ 2 }$$    
- ReLU函数  
	公式：  
	$$\begin{split}f(x)=\begin{cases} \begin{matrix} 0 & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}\end{split}$$  
	导函数：  
	$$\begin{split}{ { f }(x) }^{ ' }=\begin{cases} \begin{matrix} 0 & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}\end{split}$$  
- LeakyReLU函数  
	公式：  
	$$\begin{split} f(x)=\begin{cases} \begin{matrix} \alpha x & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}\end{split}$$  
	导函数：  
	$$\begin{split}{ { f }(x) }^{ ' }=\begin{cases} \begin{matrix} \alpha & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}\end{split}$$    
- ELU函数  
	公式：  
	$$\begin{split} f(\alpha ,x)=\begin{cases} \begin{matrix} \alpha \left( { e }^{ x }-1 \right) & x<0 \end{matrix} \\ \begin{matrix} x & x\ge 0 \end{matrix} \end{cases}\end{split}$$  
	导函数：  
	$$\begin{split}{ { f }(\alpha ,x) }^{ ' }=\begin{cases} \begin{matrix} f(\alpha ,x)+\alpha & x<0 \end{matrix} \\ \begin{matrix} 1 & x\ge 0 \end{matrix} \end{cases}\end{split}$$   
- Softplus函数  
	公式：  
	$$f(x)=\ln { (1+{ e }^{ x }) }$$  
	导函数：  
	$${ f }^{ ' }(x)=\frac { 1 }{ 1+{ e }^{ -x } }$$  
- Softsign函数  
	公式：  
	$$f(x)=\frac { x }{ \left| x \right| +1 }$$  
	导函数：  
	$${ f }^{ ' }(x)=\frac { 1 }{ { (1+\left| x \right| ) }^{ 2 } } $$  
- Swish函数  
	公式：  
	$$f\left( x \right) =x\cdot \sigma \left( x \right) $$  
	其中，$\sigma(x)$ 是 $sigmoid$ 函数。
	导函数：  
	$$f^{'}\left( x \right) =\sigma \left( x \right) +x\cdot \sigma^{'} \left( x \right) $$  

## 一些可能的想法  
- 过滤函数  
	神经网络不一定是连续输出。可以设计一个函数，将连续输出转化为离散输出。这个函数应该设置在神经网络的最后，输出结果时如果有需要就自动执行。另外，离散方法也应该存储在net类中。
	这个函数接受一个输出矩阵，如果需要离散输出，就将连续输出转化为离散输出。可以利用Softmax函数实现。    

	传入参数：
	- 前向传播函数的输出矩阵  

- 归一化  
	神经网络对于在-1至1之间的数据性能较好，因此可以对输入的数据进行处理，让数据更易于处理。也有的说法认为应当对每一层的输入进行归一化，但我认为没必要。另外，如果对预期输出也进行归一化，那么在实际预测时需要还原归一化后的数据为原始数据。  
	最常用的归一化方法是0均值归一化方法，可以将原始数据集归一化为均值为0、方差1的数据集，公式如下：  
	$$x{'}=\frac { x - \mu }{ \sigma }$$  
	其中μ为所有样本数据的均值，σ为所有样本数据的标准差。  
	显然，现在采用的一组一组输入的方法是无法进行归一化的，因此后期需要修改为多组数据同时传入。  

- 测试  
	如果只对于单组数据，可以传入输入矩阵和预期输出矩阵，只执行前向传播和损失函数就行了。但如果对于多组数据，那么可以算一点其他数据，像是准确率之类的。