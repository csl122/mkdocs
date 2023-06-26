# Deep Learning

tags:: IC, Course, ML, Uni-S10
alias:: DL
Ratio: 5:5
Time: 星期五 9:00 - 11:00

- ## Notes
	- 术语表
		- LVM: Latent Variable Model 隐变量模型
		- GMM: Gaussian Mixture Model 高斯混合模型
		- VAE: Variational Auto-encoder 变分自编码器
		- GAN: Generative Adversarial Network 生成式对抗网络
		- PCA: Principal Component Analysis 主成分分析
		- KL Divergence: Kullback-Leibler divergence KL散度
		- ELBO: evidence lower bound
	- Week2 Curse of dimensionality & Convolution & Padding & Strides
	  collapsed:: true
		- Slides
		  collapsed:: true
			- Curse of dimensionality
				- As the number of features or dimensions grows,
				  the amount of data we need to generalise accurately grows exponentially!
				- [[Lipschitz]] continuous function
					- 在[数学](https://zh.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8)中，特别是[实分析](https://zh.wikipedia.org/wiki/%E5%AF%A6%E5%88%86%E6%9E%90)，**利普希茨连续**（Lipschitz continuity）以德国数学家[鲁道夫·利普希茨](https://zh.wikipedia.org/wiki/%E9%B2%81%E9%81%93%E5%A4%AB%C2%B7%E5%88%A9%E6%99%AE%E5%B8%8C%E8%8C%A8)命名，是一个比一致[连续](https://zh.wikipedia.org/wiki/%E9%80%A3%E7%BA%8C%E5%87%BD%E6%95%B8)更强的光滑性条件。直觉上，利普希茨连续函数限制了函数改变的速度，符合利普希茨条件的函数的斜率，必小于一个称为利普希茨常数的实数（该常数依函数而定）
				- ![image.png](../assets/image_1674811744802_0.png)
				- ![image.png](../assets/image_1674208253741_0.png)
				- 例如说我们需要20%的数据来帮助KNN来决定哪个label, 1d的情况只需要0.2个单位, 但是如果是2d就需要0.45^2约等于0.2; 也就是说, 两个维度都需要0.45个单位大小来获得0.2的数据量
		- Notes
		  collapsed:: true
			- Leo Breiman 100 observations cover well the one-dimensional space of real numbers between 0 and 1.
			- To achieve a similar coverage as in one-dimensional space, 100^10 = 10^20 samples must be taken for 10-dimensional space, which results in a much higher effort.
			- Formulation of **curse**: different kinds of random distributions of the data sets, the difference between the smallest and the largest distance between data sets becomes arbitrarily small compared to the smallest distance if the dimensionality d increases (in other words, the smallest and largest distance differ only relatively little). 纬度越大, 最小距离和最大距离的差距会越小, 因为维度大上去以后, 大家都得跨越千山万水, 因此在这种情况, 普通的距离函数的作用就不大了
			- 还有一种**curse**的描述是, 高纬sphere和高纬cube之间的比值, 会随着维度的增加急剧缩小, 点和中心的距离会特别大
				- *"the volume of a high dimensional orange is mostly in its skin, not the pulp!"*
				- the volume of the hypersphere becomes very small (”insignificant”) compared to the volume of the hypercube with increasing dimension
				- 为什么高纬度对clustering method不友好: the number of dimensions increase, **points move further away from each other**, which causes *sparsity* (large unknown subspaces) and common distance measures to break down. 因此是不是可以认为dense space could still be used for clustering analysis with distance measures? #Question
				- ![image.png](../assets/image_1674210238478_0.png)
			- ![image.png](../assets/image_1674210765518_0.png)
			- Weight sharing
				- The weights of this network are sparse and spatially shared.
				- These shared weights can be interpreted as weights of a filter function. The input tensor is convoluted with this filter function (c.p. sliding window filter) before the activation of a convolutional layer.
			- Why convolution flip?
				- In NN, no flip for convenience.
				- When performing the convolution, you want the kernel to be flipped with respect to the axis along which you're performing the convolution because if you don't, you end up computing a correlation of a signal with itself.
			- [[Padding]] and [[Stride]] #convolution #[[transposed convolution]]
				- [[Convolution]]
					- Assuming padding of size p, no stride
						- $o_j=i_j-k_j+2p+1$
						- 例如原图是5, kernal是4, padding是2, output就是5-4+4+1 = 6
					- Assuming padding of size p, stride $s_j$
						- $o_j=[(i_j-k_j+2p)/s_j]+1$
						- 例如原图是5, kernal是3, padding是1, stride是2, output就是 (5 − 3 + 2)/2 + 1 = 3
				- [[Transposed Convolution]]
					- 转置卷积刚刚说了，主要作用就是起到上采样的作用。但转置卷积不是卷积的逆运算（一般卷积操作是不可逆的），它只能恢复到原来的大小（shape）数值与原来不同。转置卷积的运算步骤可以归为以下几步：
						- 在输入特征图元素间填充s-1行、列0（其中s表示转置卷积的步距）
						- 在输入特征图四周填充k-p-1行、列0（其中k表示转置卷积的kernel_size大小，p为转置卷积的padding，注意这里的padding和卷积操作中有些不同）
					- 例如:
						- 下面假设输入的特征图大小为2x2（假设输入输出都为单通道），通过转置卷积后得到4x4大小的特征图。这里使用的转置卷积核大小为k=3，stride=1，padding=0的情况（忽略偏执bias）。
							- 首先在元素间填充s-1=0行、列0（等于0不用填充）
							- 然后在特征图四周填充k-p-1=2行、列0
						- ![image.png](../assets/image_1677841056401_0.png)
					- GIF
						- ![dbb10ea62b89456ca567eb69fd31d18b.gif](../assets/dbb10ea62b89456ca567eb69fd31d18b_1677841196071_0.gif)
						- ![94191375edb942a087c54173a1dd4e75.gif](../assets/94191375edb942a087c54173a1dd4e75_1677841111552_0.gif)
						- ![dc6050f7df5042f886054f16d8e522d1.gif](../assets/dc6050f7df5042f886054f16d8e522d1_1677841177478_0.gif)
					- 计算:
					- $o = (i-1) \times s - 2p + k$
					- $H_{out​}=(H_{in​}−1)×s[0]−2×p[0]+k[0]$
					- $W_{out}​=(W_{in​}−1)×s[1]−2×p[1]+k[1]$
			-
	- Week3 Invariance & equivariance & LeNet, AlexNet, VGG
	  collapsed:: true
		- Slides
			- Invariance & equivariance
				- 不变性和等变性
				- 指的是输出随着输入的变化变不变的性质
				- CNN 拥有approximate shift equivariance, 是通过convolution layers实现的, 因为原图的移动, conv完后的feature map也会拥有同样的移动
				- CNN 拥有approximate shift invariance, 是通过pooling 和striding实现的, 因为locally max pooling的结果会是一样的, 那么输出就不会变化, 如果尺度大的话, 有解决方法例如blurring
					- But striding ignores the [[Nyquist]] sampling theorem and aliases 采样率不够, 就会出现问题
				- CNN也对deformation invariance, 例如数字3的一些小细节上面的小变动, 是不影响cnn的分类输出的
				- 传统CNN 无法做到rotation equivariance, 因为旋转以后的分布变化会导致feature map产生大的不同
			- LeNet
				- ![image.png](../assets/image_1674772072380_0.png)
				- AlexNet
					- Key modifications:
					  • Add a dropout layer after two hidden dense layers
					  (better robustness / regularization)
					  SVM
					  Softmax regression
					  • Change activation function from sigmoid to ReLu
					  (no more vanishing gradient)
					  • MaxPooling
					  • Heavy data augmentation
					  • Model ensembling
			- VGG
				- Deep and narrow = better
				- a larger number of compositions of simple functions turns out to be more expressive and more able to fit meaningful models than a small number of shallower and more complicated functions.
				- ![image.png](../assets/image_1674772249868_0.png)
		- Notes
			- The [[Nyquist]] rate
				- Samples must be taken at a rate that is twice the frequency of the highest frequency component to be reconstructed.
				- Under-sampling: sampling at a rate that is too coarse, i.e., is below the Nyquist rate.
				- Aliasing: artefacts (jaggies) that result from under-sampling.
			- 图片的frequency其实就是 gradient, edge就是gradient大的地方, 就是高频部分, 平滑区域就是低频部分, 可以通过不同频率的不断采样reconstruct
	- Week4 NiN, ResNet, Inception, BatchNorm, Activations, Loss, Augmentation, U-net
	  collapsed:: true
		- Slides
			- Parameters of Convolution layers
				- $c_i \times c_0 \times k^2$
				- computation就是前面乘上feature map的大小像素个数 $m_h \times m_w$
			- Parameters of fully connected layer connected with conv and output
				- $c \times m_w \times m_h \times n$
			- Networks in Networks
				- NiN设计在conv层中间加入MLP来获取更多conv信息
				- 实现中, MLP使用了conv 1x1来作为mlp
				- ![image.png](../assets/image_1675269160886_0.png)
				- global average pooling
					- 全局平均池化, 对于每一个channel中的所有数据取平均得到唯一的一个值, 也就是几个频道就有几个值, 而一般这个频道数会事先设定为类别数, 直接建立起了conv结果与类别的关系. 取平均是为了保留该通道全局信息, 而不是最显著信息
			- Inception (GooLeNet)
				- inception net的思想就是把该用什么样的卷积核大小个数padding stride直接交给网络去优化, 每一条path只要保证去其他的形状一致即可, 输出通道数是不一样的, 最后延通道拼接起来. 期待每个channel都能work for 特定的任务 have an architecture with different channels doing different things and the hope is one of those channels is gonna work for the cats and one is gonna work for I some birds and so on.
				- ![image.png](../assets/image_1675269431882_0.png){:height 200, :width 654}
				- ![image.png](../assets/image_1675269577123_0.png)
				- 由于是不同类型的卷积核concat, 能够大大减少由单一卷积核生成导致的大量参数问题
				- ![image.png](../assets/image_1675270290983_0.png)
				- 神经网络第一层通常会采用例如77的conv和maxpool, 其作用在于have some basic
				  amount of translation invariance and that I am able to reduce the dimensionality reasonably quickly 平移不变性, 减少特征空间大小;
				- 第二个阶段一般会做简单的33卷积和maxpool来获得总体的空间上的关联get some overall spatial correlation and then some pooling operation in the end.
				- 剩余的步骤可以理解为减少pixels, 但是让他们拥有更多高层次的有用的信息shrinking resolution but I'm also increasing the number of channels because now while I have fewer pixels they have more valuable more higher- order information that I'm going to use later on.
				- 用两个33conv代替55可以减少参数量, 但是具有相同的感受野
				- V3版本中还用了17 71 conv, you might only get for example vertically contiguous features in some way
				- 学到了什么?
					- Dense layers are computationally and memory intensive
					- 1x1 convolutions act like a multi-layer perceptron per pixel.
					- If not sure, just take all options and let the optimization decide or even learn this through trial and error (genetic algorithm, AmoebaNet)
			- BatchNorm
				- the trouble is as I'm adapting from the top down, the features are going back up, are going to change so now that last layer that was actually fairly well adapted to begin with, has to readapt now to the new inputs that it's getting. long time to converge and adopt all layers.
				- ![image.png](../assets/image_1675270979934_0.png)
				- BN在训练中, 每个batch都会以channel算出一个整个batch的mean 和variance, 比如3个通道, 就会有三个mean 和variance, 然后所有的数据根据这个值来进行normalise
				  测试中, 则使用训练过程中不断更新的running mean 和variance进行normalise, 也就是整个数据集的, 是在一个个batch iter中类似于temporal difference 更新的, 代码如下 #BatchNorm #ML
					- ```python
					  class BatchNorm2d(nn.Module):
					      def __init__(self, num_features, eps=1e-05, momentum=0.1):
					          super(BatchNorm2d, self).__init__()
					          """
					          An implementation of a Batch Normalization over a mini-batch of 2D inputs.
					  
					          The mean and standard-deviation are calculated per-dimension over the
					          mini-batches and gamma and beta are learnable parameter vectors of
					          size num_features.
					  
					          Parameters:
					          - num_features: C from an expected input of size (N, C, H, W).
					          - eps: a value added to the denominator for numerical stability. Default: 1e-5
					          - momentum: the value used for the running_mean and running_var
					          computation. Default: 0.1 . (i.e. 1-momentum for running mean)
					          - gamma: the learnable weights of shape (num_features).
					          - beta: the learnable bias of the module of shape (num_features).
					          """
					          # TODO: Define the parameters used in the forward pass                 #
					          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					          self.num_features = num_features
					          self.eps = eps
					          self.momentum = momentum
					  
					          # self.register_parameter is not used as it was mentioned on piazza
					          # that this will be overridden
					          self.gamma = torch.ones((1, num_features, 1, 1))
					          self.beta = torch.zeros((1, num_features, 1, 1))
					          self.running_mean = torch.zeros((1, num_features, 1, 1))
					          self.running_var = torch.ones((1, num_features, 1, 1))
					  
					      # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					  
					      def forward(self, x):
					          """
					          During training this layer keeps running estimates of its computed mean and
					          variance, which are then used for normalization during evaluation.
					          Input:
					          - x: Input data of shape (N, C, H, W)
					          Output:
					          - out: Output data of shape (N, C, H, W) (same shape as input)
					          """
					          # TODO: Implement the forward pass                                     #
					          #       (be aware of the difference for training and testing)          #
					          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					          if self.training:
					              mean = x.mean(dim=(0, 2, 3), keepdim=True)
					              var = x.var(dim=(0, 2, 3), keepdim=True)
					  
					              x_hat = (x-mean)/torch.sqrt(var+self.eps)
					              self.running_mean = self.momentum * mean + (1-self.momentum) * self.running_mean
					              self.running_var = self.momentum * var + (1-self.momentum) * self.running_var
					          else:
					              x_hat = (x-self.running_mean)/torch.sqrt(self.running_var+self.eps)
					  
					          x = self.gamma * x_hat + self.beta
					          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
					          return x
					  ```
				- **capacity control**: it does not really reduce covariate shift. they find out that it actually makes covariate shift worse. it actually turns out that basically this is doing regularization by noise injection. a mini batch of maybe 64 observations and so what you’re effectively doing is you’re subtracting some empirical mean and that's obviously noisy. And you’re dividing by some empirical standard deviation. That's obviously also noisy.
				- Batch Norm对batch size 敏感:
					- batch that’s too large then you're not injecting enough noise and you're not regularizing enough.
					- one that’s too small then basically the noise becomes too high and then you’re not converging very well.
					- 但是已经通过gamma和beta来修正解决了a learned scale and offset.
			- ResNet
				- ![image.png](../assets/image_1675271910997_0.png)
				- ![image.png](../assets/image_1675271930329_0.png)
				- ‘Taylor expansion’ style parametrization
				- it also means that as I add another layer, the identity function still goes through and it will still leave the outputs of the previous layers unchanged.
				- ResNet通常四个layer, 每个layer最开始的那个block因为需要对齐残差结构的输入和输出, 需要上面图中的11conv来进行频道对齐以及downsample
			- ResNext
				- ![image.png](../assets/image_1675272224692_0.png)
				- So what we are doing is basically taking the network on the left, and slicing it up into 16 or more networks that just have four channels, or some larger number of channels each. And then in the end, we stack everything together. 减少了参数个数, 通过分组的方式
				- ![image.png](../assets/image_1675272362556_0.png)
			- DenseNet
				- ![image.png](../assets/image_1675272406263_0.png)
				- ![image.png](../assets/image_1675272469553_0.png)
			- Squeeze-Excite Net
				- ![image.png](../assets/image_1675272880491_0.png)
				- So attention is essentially a mechanism where, rather than taking averages over a bunch of vectors, we're using a separate function to gate how that average should be computed. attention 定义了这个average的操作怎么做
				- SE-Net做的事情就是, 例如有很多频道, 这个频道是专攻cat的, 另一个是恐龙, 当你预测猫咪的时候肯定要weight 猫咪多一点, 所以这个网络就是在learn global weighting function per channel. 但是我们如果看到有一碗牛奶的时候, 那有猫咪的可能性就会高很多, 这就是一个思路, 把channel之间的信息链接起来
				- What you could actually do is you could take very simply a product of the entire image in a per channel basis, with some other vector. 相当于知道了关联度. Finally you use those numbers and the softmax over them to rewrite your channels.
				  
				  So therefore, if this very cheap procedure tells me, well, there's a good chance that there's a cat somewhere, I can now up weigh the cat channel.
					-
			- ShuffleNet
				- ![image.png](../assets/image_1675272976241_0.png)
				- we do something a little bit more structured, or, more sparse, structured with our networks.
				- if we have three channels, well, we go and basically pick, one from the red greens and blues and turn that into a new block.
				  And then I essentially intertwine things in a meaningful way.
				- Efficient on mobile devices
			- Activation functions
				- Linear activation
					- ![image.png](../assets/image_1675273197243_0.png)
					- if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer!
					- That means these N linearly activated layers can be replaced by a single layer.
					  No matter how we stack, the whole network is still equivalent to a single layer with linear activation
				- Sigmoid function
				  ![image.png](../assets/image_1675273291921_0.png)
					- looks smooth and “step function like”
					- has a tendency to bring the Y values to either end of the curve.
					- Making clear distinctions on prediction.
					- the Y values tend to respond very less to changes in X. What does that mean? The gradient at that region is going to be small. It gives rise to a problem of “vanishing gradients”.
					  This meant the network may refuse to learn further or is drastically slow.
				- tanh function
					- ![image.png](../assets/image_1675273339650_0.png)
					- One advantage of tanh is that we can expect the output to be close to have a zero-mean.
					- because they see positive and negative values and therefore tend to converge faster.
					- Deciding between the sigmoid or tanh will depend on your requirement of gradient strength.
					- 不能stack太多
				- ReLU
					- ![image.png](../assets/image_1675273692839_0.png)
					- It is not bound and the range of ReLu is [0, inf). This means it can blow up the activation.
					  the sparsity of the activation can also be a problem in networks.
					- We would ideally want a few neurons in the network to not activate and thereby making the activations sparse and efficient.
					  RELU allows this.
					- Because of the horizontal line in ReLu( for negative X ), the gradient can go towards 0. For activations in that region of ReLu, gradient will be 0 because of which the weights will not get adjusted during descent. That means, those neurons which go into that state will stop responding to variations in error/ input. This is called dying ReLu problem. This problem can cause several neurons to just die and not respond making a substantial part of the network passive.
					- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.
				- Leaky ReLU
					- ![image.png](../assets/image_1675273758735_0.png)
					- for example y = 0.01x for x<0 will make it a slightly inclined line rather than horizontal line. This is leaky ReLu. There are other variations too. The main idea is to let the gradient be non zero and recover during training eventually.
				- PReLU
					- ![image.png](../assets/image_1675273805475_0.png)
					- Here a is a learnable parameter.
					- What’s interesting about the ReLus is that they are scale invariant.
					  You can multiply the signal by a value and the output will not be changed, except the scale. So these are equivariant to scale. There is only one linearity.
				- SoftPlus
					- ![image.png](../assets/image_1675273849200_0.png)
					- Functions like these here are affected by the amplitude of the input signal.
					- Softplus is a smooth approximation of the ReLU function. A differentiable version of ReLU.
					- It has a scale parameter beta. The higher beta, the more this function will look like a ReLU.
					  It can be used to constrain the output of a unit to always be positive.
				- ELU
					- ![image.png](../assets/image_1675273902630_0.png)
					- Another soft version of RELU. You use Relu as a basis and add a small constant that makes it smooth.
					- One difference is that this one here can become negative, unlike the RELu.
					  That may have an advantage but very much depends on the application it is used for. Allows the network to make the average of the output zero, which can help convergence.
				- CELU
					- ![image.png](../assets/image_1675273941843_0.png)
					- ![image.png](../assets/image_1675273954685_0.png)
				- SELU
					- ![image.png](../assets/image_1675273977136_0.png)
				- GELU
					- ![image.png](../assets/image_1675273992870_0.png)
				- ReLU6
					- ![image.png](../assets/image_1675274005460_0.png)
				- LogSigmoid
					- ![image.png](../assets/image_1675274021411_0.png)
				- Softmin
					- ![image.png](../assets/image_1675274032679_0.png)
				- Softmax
					- ![image.png](../assets/image_1675274042775_0.png)
				- LogSoftmax
					- ![image.png](../assets/image_1675274053809_0.png)
				- Which function to use depends on the nature of the targeted problem. Most often you will be fine with ReLUs for classification problems. If the network does not converge, use leakyReLUs or PReLUs, etc. Tanh is quite ok for regression and continuous reconstruction problems.
			- Loss
				- ![image.png](../assets/image_1675299602450_0.png)
				- ![image.png](../assets/image_1675299610352_0.png)
				- ![image.png](../assets/image_1675299817956_0.png)
				- ![image.png](../assets/image_1675299824732_0.png)
				- ![image.png](../assets/image_1675299833192_0.png)
				- ![image.png](../assets/image_1675299838997_0.png)
				- ![image.png](../assets/image_1675299848521_0.png)
				- ![image.png](../assets/image_1675299861572_0.png)
				- ![image.png](../assets/image_1675299871627_0.png)
				- ![image.png](../assets/image_1675299879651_0.png)
					-
			- U-net
				- ![image.png](../assets/image_1675299996606_0.png)
			- Augmentation
				- ![image.png](../assets/image_1675300007063_0.png)
				- ![image.png](../assets/image_1675300014671_0.png)
		- Notes
			- mini batch的用法由来
				- 没有办法把整个datasset放进来, 只能random sample 一些, 来作为估计
				- 但是后来发现这个还有regularisation作用, 每个batch有自己的分布, 导致weights不准了
	- Week5 Generative models, VAE, GAN
		- Unsupervised learning: Given data without supervision signal, goal is to infer a function that describes the hidden structure of unlabelled data
			- • Probability distribution/density estimation
			  • Dimensionality Reduction 
			  • Clustering
		- Probability distribution/density estimation
			- ![image.png](../assets/image_1676114034268_0.png)
			- 数据是从某个分布中才养出来的, 我们的目的是学习一个分布参数, 能让这个分布模型生成出来的数据分布和原数据分布很像
		- Generative Latent Variable Models
			- ![image.png](../assets/image_1676114228274_0.png)
			- z是隐变量, 观测不到的一些潜在分布属性, 可以用来代表一类的samples, 例如可以是写的字的label, 书写风格, 相机的角度, 光照条件, x是通过这些东西决定生成的实际分布结果的
			- x是可观测变量, 即为观测到的结果, 例如生成的数字, 实际的照片
			- p(z)是先验, 需要先知道才能有后续生成, 我们假定他服从一定的概率分布, 来表示某一类的sample的先验潜在变量
			- p(x|z)是似然, 是给定z后x出现的概率
			- p(x)是evidence, 在这个问题背景下, 也是我们要拟合的对象, 通过sample z, 加起来expectation就是p(x), 也就是上图的推导式子
			- p(z|x)是后验, 是得到x以后, 某个z出现的概率, 我们希望能用这个模型sample出与x高度相关的z来帮助生成有用的x
		- Dimensionality reduction (Prob PCA)
			- ![image.png](../assets/image_1676115292249_0.png)
			- 用概率学来进行主成分分析, 对原本数据的d维度降维到k, z就是k维的可以表征一大类x的隐变量 存有绝大多数有用的信息, 因此可以用来复原x.
			- 这里第二行就是想要学习一个映射关系, 以期把k维度的z恢复成d维的x, 也就是x conditioned on z
			- z也是需要建模的, 因为我们不知道哪些z是有用的z, 是包含了主要成分的z, 而符合某一个分布的z就可能满足响应的要求, 能够被第二个式子中的W给有效恢复成可行的x
			- 生成Z的过程就是降维, encoder的过程
			- 从z生成x 的过程就是升维, decode的过程
		- Jensen's inequality:
			- ![image.png](../assets/image_1676116036467_0.png)
			- 注意: 如果是下凹函数的话, 这个不等式的大于等于就是小于等于了, 例如log
		- Divergence minimisation (KL divergence)
			- ![image.png](../assets/image_1676115811106_0.png)
			- $KL[p(x)||q(x)]\ =\ E_{p(x)}[log\frac{p(x)}{q(x)}]$
			- ![image.png](../assets/image_1676116070392_0.png)
		- Maximum Likelihood Estimation
			- ![image.png](../assets/image_1676116224094_0.png)
			- 这个简单理解就是, KL divergence其实就是让theta下的prob最接近真实情况, 所以就是MLE, 从data中采样的点, 在我们的概率模型中出现的概率最大化, 那就是很接近了
		- ((63e78193-b63d-4865-b333-a0759ae6f935))
			- ((63e78203-d202-469c-9dd8-1185c4cbb85d))
			- 我们的p(x)的建立, 需要p(z)这个先验, 但是我们没办法做到sample所有可能的z, 而且与x无关的z只会带来无用的噪音, 因此我们需要用到后验p(z|x)来找到适合的z
		- [[VAE]]
			- References: [机器学习方法—优雅的模型（一）：变分自编码器（VAE） - 知乎](https://zhuanlan.zhihu.com/p/348498294?utm_id=0)
			- ((63e782cc-b656-4ffa-8c6e-63a2ca501ad0))
			- ((63e782f0-897d-43d3-bfb4-5753ec863a3c))
			- ((63e782fd-6145-4156-b8b2-7d112b470d1c))
			- ((63e7830d-d431-463f-8f8c-2a92414de066))
			- ((63e7831c-09a9-43e3-8b78-819d4dd176c4))
				- 原本是采样z来进行训练, 没有办法进行BP, 但是我们通过epsilon这个从高斯分布中采样的东西来改变z的方差, 来获得不同的z样本, 因为epsilon是一个常量, 就不用担心BP的问题了
				- ((63e78419-fa60-4c52-91d5-cb831af8bb52))
			- 下面讲两个推出Loss的方式
			- ((63e78343-7822-49c1-8424-b2a339b4d13d))
			- ((63e78350-a414-4295-ab0f-e8d999023d5b))
			- ((63e7836c-9a5e-438a-8dc0-f782dbdb53dd))
			- VAE objective
				- ((63e78381-3db8-4b4e-9bce-7e3201dda44a))
			- 我们需要给encoder 定义一个需要学习的分布的标准 q distribution design
				- ((63e783a9-82ca-405d-8757-d0d2985b54a1))
				- 复杂的, 不同的分布假设, 会带来不一样的结果
			- ![image.png](../assets/image_1676117078580_0.png)
				- 最终的Loss考虑到了reconstruction loss, 即极大似然部分, 生成数据和数据集的相似程度, 以及KL regulariser, 用来让我们设计的encoder中的q接近我们期望的真实隐变量分布p(z)
			- Practical implementation
				- ((63e78506-afdc-486f-82f4-914c0d7608e0))
		- [[GAN]]
			- ![image.png](../assets/image_1676117573867_0.png)
			- Two-player game objective:
				- ((63e78686-63fa-4ba2-ab32-206fdbbb14fa))
				- 对Discriminator 而言, 这个Loss要越大越好, 表示他识别出来了
				- 对Generator而言, 这个Loss要越小越好, 表示不能被认出来我生成的图片
				- ![image.png](../assets/image_1676117712973_0.png)
				- 训练的时候, 固定某一个参数, 训练另一个
			- Solving the two-player game objective
				- ((63e786ff-afce-47d9-92c9-626bf5e5eef6))
			- ((63e78726-8149-4648-8cb9-534c26e34832))
				- ((63e78730-bee1-494f-8775-cc1a44382fe0))
			- Practical implementation for solving
				- ((63e78748-a626-4d85-8f35-3d9bb651073c))
			- Practical strategy for training the generator G
				- At the beginning, generated image quality is bad, Generator 可以很轻易的识别出来, 难以继续训练了
				- ((63e7879d-c35e-45b4-abca-bf9d841e2daa))
			- ((63e787a9-d297-450a-b944-35b2356bfc16))
				- ((63e787b7-222e-4a99-9cbe-9c18400d627f))
				- ((63e787d3-c2fe-41bb-99e0-91f287757609))
				- ((63e787e6-8e9a-4d1e-9b6c-8ff984a05cf9))
	- Week6 RNN, LSTM, Attention, Transformers
		-
		-
	- Week7
	- Week8
- ## CW
  collapsed:: true
	- GAN
	    * Enlarge the model with more layers and parameters
	    * learning rate scheduler
	    * In GAN, if the discriminator depends on a small set of features to detect real images, the generator may just produce these features only to exploit the discriminator. The optimization may turn too greedy and produces no long term benefit. In GAN, overconfidence hurts badly. To avoid the problem, we penalize the discriminator when the prediction for any real images go beyond 0.9 (D(real image)>0.9). This is done by setting our target label value to be 0.9 instead of 1.0.
	    * larger learning rate for the discriminator
	    * Two Time-Scale Update Rule
	    *  Add noise to the real and generated images before feeding them into the discriminator.
	    *  当GAN生成的图像不够准确、清晰时，可尝试增加卷积层中的卷积核的大小和数量，特别是初始的卷积层。卷积核的增大可以增加卷积的视野域，平滑卷积层的学习过程，使得训练不过分快速地收敛。增加卷积核数（特别是生成器），可以增加网络的参数数量和复杂度，增加网络的学习能力。
	    *  Real=False, fake=True
	    *  leaky for generator
	    *  transfer from VAE
	    *  在dis的卷前加入高斯噪声
	    *  new loss function
	- * What didn't work, what worked and what mattered most
	    * Add noise to the real and generated images before feeding them into the discriminator.
	    * 在输入图片中加入高斯噪声
	    * dropout
	  
	  * Are there any tricks you came across in the literature etc. which you suspect would be helpful here
	    * Wasserstein GANs
- ## PDF #PDF
  collapsed:: true
	- ![N16a_VAE_Notes.pdf](../assets/N16a_VAE_Notes_1676113337238_0.pdf)
	- ![N17a_GAN_Notes.pdf](../assets/N17a_GAN_Notes_1676117464510_0.pdf)
	- ![N16_VAE 2.pdf](../assets/N16_VAE_2_1676849158717_0.pdf)
	- ![N17_GAN 2.pdf](../assets/N17_GAN_2_1676849171594_0.pdf)
	- ![N18_RNN 2.pdf](../assets/N18_RNN_2_1676849179285_0.pdf)
	- ![N20_attention 2.pdf](../assets/N20_attention_2_1676849185672_0.pdf)
	-
- ## Info
  collapsed:: true
	- 5:5
	- 星期五 9:00 - 11:00
	- CW: 2 Tasks, both assessed, Task 1: 40%, Task 2: 60%
		- [Cloud computing, evolved | Paperspace](https://www.paperspace.com/)
		- [Google Colab](https://colab.research.google.com/)
		- 答案在两周后公布
	- Tutorial: 每周都有, 答案周一公布
- ## Syllabus
  collapsed:: true
	- Supervised vs unsupervised learning, generalisation, overfitting
	- Perceptrons, including deep vs shallow models
	- Stochastic gradient descent and backpropagation
	- Convolutional neural networks (CNN) and underlying mathematical principles
	- CNN architectures and applications in image analysis
	- Recurrent neural networks (RNN), long-short term memory (LSTM), gated recurrent units (GRU)
	- Applications on RNNs in speech analysis and machine translation
	- Mathematical principles of generative networks; variational autoencoders (VAE); generative adversarial networks (GAN)
	- Applications of generative networks in image generation
	- Graph neural networks (GNN): spectral and spatial domain methods, message passing
	- Applications of GNNs in computational social sciences, high-energy physics, and medicine
	- • Feature extraction, convolutions and CNNs
	  • Common Network architectures
	  • Automatic parameter optimisation
	  • RNNs, LSTMs, GRUs
	  • VAEs and GANs
	  • GNNs
	  • Deep learning programming frameworks • Applications of deep learning
## Links
collapsed:: true
	- [Scientia](https://scientia.doc.ic.ac.uk/2223/modules/70010/materials)
	- [70010 Deep Learning | Bernhard Kainz](http://wp.doc.ic.ac.uk/bkainz/teaching/70010-deep-learning/)
	- [Panopto](https://imperial.cloud.panopto.eu/Panopto/Pages/Sessions/List.aspx#folderID=%228631beb3-a0b0-4b5d-b85e-aedd01736de2%22&sortColumn=0&sortAscending=true)
	- [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation](https://zh.d2l.ai/) #书 #科技资源 #ML #DL #tutorial
	- [Rent GPUs | Vast.ai](https://vast.ai/)
	- [Cloud computing, evolved | Paperspace](https://www.paperspace.com/) #科技资源 #GPU #服务器
	- [Browse the State-of-the-Art in Machine Learning | Papers With Code](https://paperswithcode.com/sota/)
	- #科技资源 #GitHub #ML #DL #Academic
		- • [GitHub - alievk/avatarify-python: Avatars for Zoom, Skype and other video-conferencing apps.](https://github.com/alievk/avatarify)
		  • [GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model](https://github.com/CompVis/stable-diffusion)
		  • [GitHub - deepfakes/faceswap: Deepfakes Software For All](https://github.com/deepfakes/faceswap)
		  • [GitHub - Avik-Jain/100-Days-Of-ML-Code: 100 Days of ML Coding](https://github.com/Avik-Jain/100-Days-Of-ML-Code)
		  • [GitHub - facebookresearch/detectron2: Detectron2 is a platform for object detection, segmentation and other visual recognition tasks.](https://github.com/facebookresearch/Detectron2)
		  • [GitHub - fastai/fastai: The fastai deep learning library](https://github.com/fastai/fastai)
		  • [GitHub - CMU-Perceptual-Computing-Lab/openpose: OpenPose: Real-time multi-person keypoint detection library for body, face, hands, and foot estimation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
		  • [GitHub - matterport/Mask_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN)
		- [GitHub - soumith/ganhacks: starter from "How to Train a GAN?" at NIPS2016](https://github.com/soumith/ganhacks#authors) #GAN
		-
-
-
