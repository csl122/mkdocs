# Machine Learning for Imaging

tags:: IC, Course, ML, Uni-S10
alias:: MLI
Ratio: 7:3
Time: 星期五 14:00 - 16:00

- ## Notes
	- Week2 intro
	  collapsed:: true
		- Lecture
		- Tutorial
		  collapsed:: true
			- 基本的机器学习类, 以regresion为例
			- ```python
			  
			  class LogisticRegression:
			      def __init__(self, lr=0.05, num_iter=1000, add_bias=True, verbose=True):
			          self.lr = lr
			          self.verbose = verbose
			          self.num_iter = num_iter # 多少个epoch
			          self.add_bias = add_bias # 用于加入bias
			          self.weight = np.random.normal(0, 0.01, 50) # 如果知道多少参数的话, 可以直接初始化, 
			          # 也可以fit里面根据实际feature数量决定
			      
			      def __add_bias(self, X):
			          bias = np.ones((X.shape[0], 1)) # (10000, 1) 多加了一个bias 放在原本的feature之后, 用于求和的时候多一个bias
			          return np.concatenate((bias, X), axis=1) # 多加一个列维度
			      
			  
			  	# 损失函数
			      def __loss(self, h, y):
			          ''' computes loss values '''
			          y = np.array(y,dtype=float)
			          ############################################################################
			          # Q: compute the loss 
			          ############################################################################
			          return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() # 二分类问题的cross-entropy loss, 两个分类分别的cross entropy相加, 因为是二分类, 所以是y, 1-y
			        # 总和为1的类概率, 用上cross entropy, 对权重求导的结果就是1/N (y^ - y)
			  
			      
			      def fit(self, X, y):
			          ''' 
			          Optimise our model using gradient descent
			          Arguments:
			              X input features
			              y labels from training data
			              
			          '''
			          if self.add_bias:
			              X = self.__add_bias(X)
			          
			          ############################################################################
			          # Q: initialise weights randomly with normal distribution N(0,0.01)
			          ############################################################################
			          self.theta = np.random.normal(0.0,0.01,X.shape[1])
			          
			          for i in range(self.num_iter):
			              ############################################################################
			              # Q: forward propagation
			              ############################################################################
			              z = X.dot(self.theta) # 乘上了权重后的结果， 每个feature 都有一个对应的权重，乘起来相加就是某个sample的值
			              h = 1.0 / (1.0 + np.exp(-z)) # activation, 得到预测概率
			              ############################################################################
			              # Q: backward propagation
			              ############################################################################
			              # (h - y) / y.size 是 Loss对z的求导; X则是z对weight的求导, 因为z
			              gradient = np.dot(X.T, (h - y)) / y.size # (785,) 11774个sample， 每个维度都是所有sample梯度的和， 最后除以了sample 量
			              # update parameters 梯度下降
			              self.theta -= self.lr * gradient
			              ############################################################################
			              # Q: print loss
			              ############################################################################
			              if(self.verbose == True and i % 50 == 0): 
			                  h = 1.0 / (1.0 + np.exp(-X.dot(self.theta)))
			                  print('loss: {} \t'.format(self.__loss(h, y)))
			      
			      def predict_probs(self,X):
			          ''' returns output probabilities
			          '''
			          ############################################################################
			          # Q: forward propagation
			          ############################################################################
			          if self.add_bias:
			              X = self.__add_bias(X)
			          z = X.dot(self.theta)
			          return 1.0 / (1.0 + np.exp(-z))
			  
			      def predict(self, X, threshold=0.5):
			          ''' returns output classes
			          '''
			          return self.predict_probs(X) >= threshold
			  
			  model = LogisticRegression(lr=1e-2, num_iter=1000)
			  # shuffle data
			  shuffle_index = np.random.permutation(len(selected_labels))
			  selected_data, selected_labels = selected_data[shuffle_index], selected_labels[shuffle_index]
			  model.fit(selected_train_data, selected_train_labels)
			  train_preds = model.predict(selected_train_data)
			  logistic_train_acc = 100.0 * (train_preds == selected_train_labels).mean()
			  ```
	- Week3 Classification
	  id:: 63dacd04-711a-4465-a450-60bf0076ed43
	  collapsed:: true
		- Lecture
			- feature extraction
			  collapsed:: true
				- pipeline
				  collapsed:: true
					- images->features->prediction
				- feature extraction and descriptors
				  collapsed:: true
					- – Intensities
					  collapsed:: true
						- ![image.png](../assets/image_1674664156563_0.png)
					- – Gradient
					  collapsed:: true
						- Gradients: Invariant to absolute illumination
					- – Histogram
						- patch-level和image-level都可以用hist
					- – SIFT
					  collapsed:: true
						- ![image.png](../assets/image_1674664387996_0.png)
					- – HoG Histogram of gradients
					  collapsed:: true
						- ![image.png](../assets/image_1674664443665_0.png)
						- ![image.png](../assets/image_1674664461115_0.png)
						- ![image.png](../assets/image_1674664760075_0.png)
					- – SURF
					  collapsed:: true
						- ![image.png](../assets/image_1674664774363_0.png)
						- ![image.png](../assets/image_1674664807093_0.png)
					- – BRIEF
					  collapsed:: true
						- ![image.png](../assets/image_1674664845583_0.png)
						-
					- – LBP
					  collapsed:: true
						- ![image.png](../assets/image_1674664927209_0.png)
					- – Haar
					  collapsed:: true
						- ![image.png](../assets/image_1674664941210_0.png)
						- ![image.png](../assets/image_1674664975489_0.png)
			- image classification: Ensemble classifiers
			  collapsed:: true
				- classification models
				  collapsed:: true
					- – Logistic regression D
					  – Naïve Bayes G
					  – K-nearest neighbors (KNN) G
					  – Support vector machines (SVM) D
					  – Boosting 
					  – Decision/Random forests D
					  – Neural networks D
				- bias and variance
				  collapsed:: true
					- Bias error
					  – how much, on average, **predicted values are different from the actual value**.
					  – high bias error means we have an under-performing model, which misses important trends.
					  Accurate和inaccurate
					- Variance error
					  – how much **predictions made from different samples vary from one other**.
					  – high variance error means the model will over-fit and perform badly on any observation beyond training.
					- variable 和consistent
					- ![image.png](../assets/image_1674665312313_0.png)
					-
				- Ensemble learning
				  collapsed:: true
					- Aggregate the predictions of a group of predictors (either classifiers or regressors) 聚合一堆预测的结果, 一堆预测器就是一个ensemble
					- A learning algorithm which uses multiple models, such as classifiers or experts, is called Ensemble Learning (so called meta-algorithms) 一个使用了多个分类器的算法就叫做ensemble learning 或者meta 算法
					- Types of ensemble learning
					  collapsed:: true
						- • Homogenous:
						  – Combine predictions made from models built from the same ML class
						  同质的, 用了同一种模型
						  比如很多weak learner, 用了不同subset of data
						- • Heterogenous:
						  – Combine predictions made from models built from different ML classes
						  不同的种类的模型
						- • Sequential – base (ML) learners are added one at a time; mislabelled
						  examples are upweighted each time
						  – Exploits the dependence between base learners – thus learning a complementary set of predictors that reduce the bias and increase the accuracy.
						  序列化的, 减小bias, 因为一层一层筛选, 会非常精确
						- • Parallel – many independent base learners are trained
						  simultaneously and then combined
						  – Combines prediction learnt from multiple models run independently averaging away impact of isolated errors - thus reduces variance of the prediction.
						  平行的, 减小方差, 减小孤立的错误的影响
					- Decision Stump
					  collapsed:: true
						- ![image.png](../assets/image_1674666376082_0.png)
					- Ensemble methods
						- • Voting
						- • Bagging (Bootstrap Aggregation)
						  collapsed:: true
							- 平行的一种ensemble learning, 通过一堆weak learner平行预测, 来减小variance, 因为$var(x^-) = var(x)/n$
							- Bootstrapping
							  collapsed:: true
								- 1. Take the original dataset E with N training samples
								  2. Create T copies  by sampling with replacement
								      – Each copy will be different since some examples maybe repeated while others will not be sampled at all
								  3. Train a separate weak learner on each Bootstrap sample
							- Aggregating results
							  collapsed:: true
								- ![image.png](../assets/image_1674666835882_0.png)
							- Out-of-Bag (OOB) error
								- 如何评价error呢, 我们用left-out-set来作为validation set
						- • Boosting
						  collapsed:: true
							- 序列化的, 一个learner接着一个learner, 大大降低bias
							- Rather than building independent weak learners in parallel and aggregating at end:
							  – build weak learners in serial
							  – but adaptively reweight training data prior to training each new weak learner in order to give a higher weight to previously misclassified examples
							- 不断重新weight training data 来给新的weak learner, 给misclassified 更高权重
							- ![image.png](../assets/image_1674668364470_0.png)
							- Adaboost - 一种自适应的boosting, adaptive
								- ![image.png](../assets/image_1674668416628_0.png)
						- • Random Forests
						  collapsed:: true
							- Single Decision Trees are prone to overfitting, but robustness can be significantly increased by combining trees in ensembles
							- Use decision trees for homogenous ensemble learning
							- Random forests form an ensemble of uncorrelated classifiers by
							  exploiting random subsampling of the 
							  – training data used to build each tree
							  – set of features that are used for selection of the splits 
							  – set of feature values that are used for splitting
							- ![image.png](../assets/image_1674686267545_0.png)
							-
			- NN
			  collapsed:: true
				- Single-layer perceptron
				  collapsed:: true
					- ![image.png](../assets/image_1674686334440_0.png)
				- Neural networks (multilayer perceptron)
				  collapsed:: true
					- ![image.png](../assets/image_1674686588881_0.png)
				- Neural networks with K classes
				  collapsed:: true
					- one-hot encoding
						- ![image.png](../assets/image_1674686671846_0.png)
					- softmax
						- ![image.png](../assets/image_1674686779670_0.png)
					- cross entropy: Distance between probability distributions
						- ![image.png](../assets/image_1674686805554_0.png)
						- softmax和交叉熵的梯度很简单, 就是1/N * (y^ - y)
					- back propogation
						- ![image.png](../assets/image_1674687354916_0.png)
						- 往回走遇到岔路, 把当前的梯度分配出去
						- 往回走遇到汇合点, 汇合点的梯度是分叉出去的梯度和
						- ![image.png](../assets/image_1674687623328_0.png)
			- Activation and optimisation
			  collapsed:: true
				- ![image.png](../assets/image_1674687714289_0.png)
				- Optimizing neural networks: Stochastic gradient descent
					- ![image.png](../assets/image_1674687748304_0.png)
					- Large batches provide a more accurate estimate of the gradient but with less than linear returns.
					- 考虑memory, 限制了batch最大size
					- Small batch sizes can have the effect of regularization (more about regularization later) as it adds noise to the learning process. (The generalisation ability is often best with a batch size of 1, but small batch sizes require small learning rates and can lead to slow convergence).
					- 小batch会给最佳的正则化, 更好的普适性, 但是需要更小的学习率, 学习会很慢
				- gradient and learning rate
					- ![image.png](../assets/image_1674687940771_0.png)
			- tips and tricks
			  collapsed:: true
				- data augmentation
					- ![image.png](../assets/image_1674687996227_0.png)
				- Under- and overfitting
					- underfitting
						- adding more neurons, adding more layers
					- overfitting
						- adding more regularization
				- Regularisation
					- Early stopping: Interrupt training when its performance on the validation set starts dropping
					- ![image.png](../assets/image_1674688096633_0.png)
					- Max-norm
						- ![image.png](../assets/image_1674688127437_0.png)
					- Dropout
						- At every training step, every neuron (input or hidden) has a probability p of being temporarily “dropped out”:
				- Weight initialisation
					- ![image.png](../assets/image_1674688253094_0.png)
				- Normalisation
					- Standardization 标准化
						- 减去均值, 除以标准差 得到一个均值为零, 标准差为1的分布
					- 归一化
						- (X-min)/(max-min), 得到一个最大值和最小值为1和-1的缩放分布
					- Batch normalisation
					  collapsed:: true
						- ![09AF0F45-7CAD-465D-A6AC-65F3F0BEBBDB.jpeg](../assets/09AF0F45-7CAD-465D-A6AC-65F3F0BEBBDB_1674658218903_0.jpeg)
						- 每一个featuer维度, 对于一个batch里的所有sample进行标准化
						- 在NLP中会受制于每个sample句子长度不一样, 高度就不一样, 会unstable
						- ![image.png](../assets/image_1674688305892_0.png)
						- ![image.png](../assets/image_1674688422555_0.png)
					- Layer normalisation
						- 对每一个sample的所有feature做标准化
						- NLP中一个batch是有很多不同长度的句子组成的, 每个sample可以认为是多个词语的组合, 词语的数量不一定
						- NLP中就是对某个词语的整个vector做标准化
			- CNN
			  collapsed:: true
				- ![image.png](../assets/image_1674688501664_0.png)
				- 计算
					- Assuming padding of size p, no stride
						- $o_j=i_j-k_j+2p+1$
						- 例如原图是5, kernal是4, padding是2, output就是5-4+4+1 = 6
					- Assuming padding of size p, stride $s_j$
						- $o_j=[(i_j-k_j+2p)/s_j]+1$
						- 例如原图是5, kernal是3, padding是1, stride是2, output就是 (5 − 3 + 2)/2 + 1 = 3
				- 输入图有三个channel, 这个时候我们一个filter也得有三个channel和他们一一对应, 如果我们有四个filter, 那么最后生成的输出也有四个
				- pooling:
					- • Reduce size (memory) of deeper layers
					  • Invariance to small translations
					  • Contraction forces network to learn high-level features
				- upsampling
					- 把原图先填上0, 再用卷积扩大
					- ![image.png](../assets/image_1674689023013_0.png)
		- Tutorial
		  collapsed:: true
			- Haar features
				- quickly computed by integral images
				- Computational complexity of computing haar features from integral images is constant
			- End-to-end learning
				- model 能够直接从图到结果
				- 不适合小数据集, 因为需要大数据来generalise model
				- 不需要hand crafted features
				- not efficient in terms of model parameters, 因为我们需要足够复杂的模型
				- models are not human readable
			- Regularisation
				- L1, L2
					-
				- dropout
				- small batch size + SGD
				- data augmentation
	- Week4 Image Segmentation
	  collapsed:: true
		- Lecture
			- Semantic Segmentation
				- In semantic segmentation, a segmented region is also assigned a
				  semantic meaning
			- Application
			  collapsed:: true
				- conducting quantitative analyses, e.g. measuring the volume of the
				  ventricular cavity.
				- determining the precise location and extent of an organ or a certain type of tissue, e.g. a tumour, for treatment such as radiation therapy.
				- creating 3D models used for simulation, e.g. generating a model of an abdominal aortic aneurysm for simulating stress/strain distributions.
			- challenges
			  collapsed:: true
				- noise
				- partial volume effects
				  collapsed:: true
					- coarse sampling, 采样导致的不足
					- ![image.png](../assets/image_1675439370714_0.png)
				- intensity inhomogeneities, anisotropic resolution
				  collapsed:: true
					- 不同质的intensity和不同方向的resolution的不一致
					- ![image.png](../assets/image_1675439440151_0.png)
					- ![image.png](../assets/image_1675439452285_0.png)
				- imaging artifacts
					- 一些奇怪的东西导致的怪东西
				- limited contrast
					- 图片的对比度不够高
				- morphological variability,
					- 形态学上器官都长得不一样, 先验知识很难利用好
			- Evaluating Image Segmentation
				- ground truth
					- Reference or standard against a method can be compared, e.g. the optimal transformation, or a true segmentation boundary, 不现实
					- 可能人们会make up phantoms
				- gold standard
					- Experts manually segment
					- ![image.png](../assets/image_1675439700146_0.png)
			- performance measures
				- ![image.png](../assets/image_1675439726743_0.png)
				- Confusion matrix #metrics
					- ![image.png](../assets/image_1675439766890_0.png)
				- Accuracy, Precision, Recall (sensitivity), Specificity #metrics
					- ![image.png](../assets/image_1675439825709_0.png)
					- F1 score #metrics
						- ![image.png](../assets/image_1675440141000_0.png)
					- Overlap
						- Jaccard Index(IoU)
						- ![image.png](../assets/image_1675440231460_0.png)
						- Dice's Coefficient (Sorensen Index) (DSC) (Dice Similarity Coefficient)
						  collapsed:: true
							- ![image.png](../assets/image_1675440410629_0.png)
							- ![image.png](../assets/image_1675441381926_0.png)
							- ![image.png](../assets/image_1675441399747_0.png)
					- Volume similarity
					  collapsed:: true
						- ![image.png](../assets/image_1675441496345_0.png)
					- Hausdorff distance
						- ![image.png](../assets/image_1675441591113_0.png)
						- A上每个点找到B上离这个点的最短距离, 这些距离之中最长的就是我们求的这个距离
					- Symmetric average surface distance
						- ![image.png](../assets/image_1675441634954_0.png)
						- A到B的最短距离的平均数
					-
				- Segmentation Algorithms & Techniques
					- ▪ Intensity-based segmentation
					  collapsed:: true
					  ▪ e.g., thresholding
						- 用于很明显能区分主体和背景的情况
						- UL Thresholding: Select a lower and upper threshold
						- Advantages
						  ▪ simple 
						  ▪ fast
						- Disadvantages
						  ▪regions must be homogeneous and distinct
						  ▪difficulty in finding consistent thresholds across images 
						  ▪leakages, isolated pixels and ‘rough’ boundaries likely
					- ▪ Region-based
					  collapsed:: true
					  ▪ e.g., region growing
						- Start from (user selected) seed point(s), and grow a region according to an intensity threshold, 从一个点开始万物生长
						- Advantages
						  ▪relatively fast
						  ▪yields connected region (from a seed point)
						- Disadvantages
						  ▪regions must be homogeneous ▪leakages and ‘rough’ boundaries likely ▪requires (user) input for seed points
					- ▪ Graph-based segmentation
					  collapsed:: true
					  ▪ e.g., graph cuts
						- ![image.png](../assets/image_1675532483329_0.png)
						- ![image.png](../assets/image_1675532497544_0.png)
						- ![image.png](../assets/image_1675532523722_0.png)
						- Advantages
						  ▪ accurate
						  ▪reasonably efficient, interactive
						- Disadvantages
						  ▪semi-automatic, requires user input 
						  ▪difficult to select tuning parameters
					- ▪ Active contours
					  ▪ e.g., level sets
					- ▪ Atlas-based segmentation
					  collapsed:: true
					  ▪ e.g., multi-atlas label propagation
						- 基于图谱的图像分割
						- 所谓“地图集”，直观来说，Atlas 就是人工标记完备的数据库。比如 BrainWeb 的 Atlas：在三维脑部CT数据中医生标注完备的各种脑部结构，如灰质、白质、海马等等结构。
						- 将 testing image 和当前 Atlas 内的数据进行配准，然后用 Label Propagation 方法将 Atlas 数据的 Label 通过 registration mapping corresponding 关系传递到 testing image 中，从而完成 testing image 的分割任务。
						- Segmentation using registration
						- ![image.png](../assets/image_1675533840911_0.png)
						- 通过从地图集里人工标注的segmentation与样本进行映射, 这个过程叫做registration
						- Multi-Atlas Label Propagation
						- ![image.png](../assets/image_1675534069463_0.png)
						- Advantages
						  ▪robust and accurate (like ensembles) 
						  ▪yields plausible segmentations
						  ▪fully automatic
						- Disadvantages
						  ▪computationally expensive
						  ▪cannot deal well with abnormalities 
						  ▪not suitable for tumour segmentation
					- ▪ Learning-based segmentation
					  collapsed:: true
					  ▪ e.g., random forests, convolutional neural networks
						- ![image.png](../assets/image_1675534349869_0.png)
						- 提取图像中的patches, 用树中的判断方法来判断
						- Advantages
						  ▪ensemble classifiers are robust and accurate 
						  ▪computationally efficient
						  ▪fully automatic
						  Disadvantages
						  ▪shallow model, no hierarchical features 
						  ▪no guarantees on connectedness
				- CNNs for Image Segmentation
				  collapsed:: true
					- Segmentation via Dense Classification
					- 利用到了3D卷积
					- ![image.png](../assets/image_1675535246807_0.png)
					- ![image.png](../assets/image_1675535121849_0.png)
					- 抽取一个patch, 通过整体信息来学习确认中央像素的类别
					- 通过sliding window 来遍历图片, 找到每个点的类别, 非常的不高效
					- Fully Connected to Convolution: 用conv替代fc
						- ![image.png](../assets/image_1675535590111_0.png)
						- 通过共享权重, 大大减少了参数量
						- ![image.png](../assets/image_1675535748846_0.png)
						- 1x1是class label, 但是9x9还是feature map
				- Encoder-Decoder Networks
				  collapsed:: true
					- ![image.png](../assets/image_1675535864049_0.png)
				- U-Net
				  collapsed:: true
					- ![image.png](../assets/image_1675535884207_0.png)
					- 通过原图补充原信息, 下采样再上采样学习如何产生segmentation map
				- Upsampling and transpose convolution
				  collapsed:: true
					- ![image.png](../assets/image_1675536138599_0.png)
					- ![image.png](../assets/image_1675536154098_0.png)
					- ![image.png](../assets/image_1675536226508_0.png)
				- Going Deeper
				  collapsed:: true
					- Just adding more layers is inefficient (too many parameters) 
					  ▪Idea: Use only layers with small kernels
				- Multi-scale processing
				  collapsed:: true
					- How can we make the network to “see” more context
					  ▪Idea: Add more pathways which process downsampled images
					- 如何让网络看到更多上下文信息呢, 增加pathway, 两条路给神经网络走, 一个高分辨率的大图, 一个低分辨率的小图, concat起来的时候就有了更大的视野, 新增的信息增强了localisation capability
					- ![image.png](../assets/image_1675536363791_0.png)
					- ![image.png](../assets/image_1675536435030_0.png)
					- 例如上面的高分辨率的解剖学信息, 和下面的位置信息
	- Week5 Image Registration, 坐标系, transformation, intensity-based image registration, NN for regi
	  collapsed:: true
		- Image Registration: 图像配准, 就是将不同的照片align起来, 匹配起来对应的位置
			- Establish spatial correspondences between images
			- 其实就是找到两张照片对应的点, 算得transformation的方式, 进行匹配对应, 例如以前学过的image warping, 就是扭曲图片让两张照片拼接起来, 那个时候用的是特征点对应算得fundamental matrix, 现在用的方法可能倾向于用intensity 对应, 然后算不相似度, 慢慢调整. 相对于线性的transformation, 这里还用到了deformation, 会有更高无限维度的degree of freedom
		- Coordinate system
		  collapsed:: true
			- ![image.png](../assets/image_1676133104628_0.png)
		- Transformation Models
		  collapsed:: true
			- ![image.png](../assets/image_1676133128197_0.png)
			- ![image.png](../assets/image_1676133167133_0.png)
			-
				- ![image.png](../assets/image_1676133183172_0.png)
			- Free-form Deformation
				- ![image.png](../assets/image_1676134427405_0.png)
		- Applications
		  collapsed:: true
			- Satellite Imaging, Point Correspondences, Panoramic Image Stitching, Optical Flow
			- ![image.png](../assets/image_1676134672345_0.png)
		- Intra-subject Registration
		  collapsed:: true
			- ![image.png](../assets/image_1676134707810_0.png)
			- 同一个物体的registration, 对应好以后就能找到不一样的地方, 就有可能是病变组织
		- Inter-subject Registration
		  collapsed:: true
			- ![image.png](../assets/image_1676134746307_0.png)
			- 目标是构建一个平均的atlas, 我们有一堆subjects, 取自不同的人的脑子的MR
			- 首先我们对这些图片大致对齐, 进行一个avg的操作, 作为target image
			- 然后用其他的图片对这个target进行registration, 对齐后重新avg, 获得更清晰的图片, 不断重复
			- Iterative registration 
			  1) Rigid
			  2) Affine
			  3) Nonrigid
		- Segmentation using Registration
			- Atlas是地图集, 已经手动标注了label
			- 我们可以通过对atlas进行registration, map到想要进行新的标注的图片上, 就可以propagatelabel到需要segmentation的图片上去了
		- Multi-Atlas Label Propagation
		  collapsed:: true
			- ![image.png](../assets/image_1676135510352_0.png)
			- 用多个atlas对目标图片进行配准, 得到对应的segmentation label, 然后fusion 得到一个更准的结果
		- Intensity-based Registration
			- Estimation of transformation parameters is driven by the appearance of the images
			- Images are registered when they appear similar
			- ![image.png](../assets/image_1676135634245_0.png)
			- 就是根据像素的值, 进行相似度检查, 相似度最高的时候就是配准准确率高的时候
			- Objective function (cost, energy)
			  collapsed:: true
				- ![image.png](../assets/image_1676135773999_0.png)
				- moving image 通过transformation得到的图片, 与目标图片的不相似度就是这个transformation的cost
			- Optimisation problem
			  collapsed:: true
				- ![image.png](../assets/image_1676135831878_0.png)
			- Mono-modal vs Multi-modal
				- Mono-modal registration
				  ▪ Image intensities are related by a (simple) function
				  像素值可能只和单一简单的function 有关系
				- Multi-modal registration
				  ▪ Image intensities are related by a complex function or statistical relationship
				  有复杂的函数关系, 或者有统计关系
			- (Dis)similarity Measures
				- Sum of squared differences (SSD)
					- ![image.png](../assets/image_1676135932507_0.png)
					- 假设的是identity relationship, 用于mono model, 例如CT-CT
				- Sum of absolute differences (SAD)
					- ![image.png](../assets/image_1676135958621_0.png)
					- 假设的是identity relationship, 用于mono model, 例如CT-CT
				- Correlation coefficient (CC)
					- ![image.png](../assets/image_1676135998373_0.png)
					- 假设的是linear relationship, 用于mono model, 例如MR-MR
				- Statistical relationship
					- ![image.png](../assets/image_1676136107566_0.png)
					- 对应intensity的点会形成集群对应的mapping 关系
					- ![image.png](../assets/image_1676136145591_0.png)
					- 单modal就是y=x的关系
					- ![image.png](../assets/image_1676136167495_0.png)
					- multi-modal 有一些其他的对应关系
					- ![image.png](../assets/image_1676136229289_0.png)
					- [[Joint Entropy]]可以对对应点的一起出现概率建概率模型, 这个系统越稳定, 这个概率模型对应的熵肯定越小
					- ![image.png](../assets/image_1676136293852_0.png)
					- 通过到考虑本身像素点的熵, 可以得到mutual information, 一张图能被另一张图描述的可能性
					- ![image.png](../assets/image_1676136343015_0.png)
					- [[NMI]]: 更高阶且实用的方法是Normalised mutual information, 用joint entropy作为normalisation
					- ![image.png](../assets/image_1676136418948_0.png)
					- 建立的假设是统计学上的关系, 因此可以用于多模态了, 不是那么适用于单模态, 可能无法发现想要的difference
				- Multi-scale, hierarchical Registration
				  collapsed:: true
					- 配准过程中会出现的问题
					  collapsed:: true
						- Image Overlap: 仅对overlapping 部分做相似度检查, 重合部分要足够大才好用
						- Capture Range: 在capture range中才是有效的相似度检查, 分离开来了比如匹配空气, 那肯定都是100%match
							- ![image.png](../assets/image_1676136663474_0.png)
						- Local Optima
							- ![image.png](../assets/image_1676136673974_0.png)
					- 用于解决上述问题, 使用多尺度的高斯金字塔
						- ▪Successively increase degrees of freedom 
						  ▪Gaussian image pyramids
						- ![image.png](../assets/image_1676136735499_0.png)
				- Interpolation
					- ![image.png](../assets/image_1676136769578_0.png)
					- Translate 后的值不是一一对应的, 需要差值找到
				- Registration as an Iterative Process
					- ![image.png](../assets/image_1676136804437_0.png)
					- Strategies: ▪ Gradient-descent ▪Stochastic optimisation ▪Bayesian optimisation ▪Discrete optimisation ▪Convex optimisation
					  ▪ Downhill-simplex
				-
			-
		- Registration with Neural Networks
		  collapsed:: true
			- FlowNet: Learning Optical Flow with Convolutional Networks
				- ![image.png](../assets/image_1676136881616_0.png)
				- ![image.png](../assets/image_1676136889351_0.png)
				- ![image.png](../assets/image_1676136898878_0.png)
			- FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
				- ![image.png](../assets/image_1676136931367_0.png)
				- 通过一个个网络block, correct前面的结果
			- Nonrigid Image Registration Using Multi-scale 3D CNNs
				- ![image.png](../assets/image_1676136997348_0.png)
				- 多尺度的应用
			- Spatial Transformer Networks (STN)
				- [Spatial Transformer Networks Tutorial — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html) #PyTorch #Python #tutorial #ML
				- STN引入了一个新的可学习的空间转换模块，它可以使模型具有空间不变性。这个可微分模块可以插入到现有的卷积结构中，使神经网络能够在Feature Map本身的条件下自动地对特征进行空间变换，而无需任何额外的训练监督或优化过程的修改。
				- `Localisation net`模块通过CNN提取图像的特征来预测变换矩阵θ \thetaθ
				- `Grid generator`模块就是利用`Localisation net`模块回归出来的θ \thetaθ参数来对图片中的位置进行变换，输入图片到输出图片之间的变换，需要特别注意的是这里指的是图片像素所对应的位置。
				- `Sampler`就是用来解决`Grid generator`模块变换出现小数位置的问题的。针对这种情况，`STN`采用的是`双线性插值(Bilinear Interpolation)`
				- ![image.png](../assets/image_1676137528984_0.png)
			- End-to-End Unsupervised Deformable Image Registration with a CNN
				- ![image.png](../assets/image_1676137622449_0.png)
			- An Unsupervised Learning Model for Deformable Image Registration
				- ![image.png](../assets/image_1676137665251_0.png)
			- Supervised below
			  collapsed:: true
				- VoxelMorph: A Learning Framework for Deformable Image Registration
					- ![image.png](../assets/image_1676137693469_0.png)
				- Learning Conditional Deformable Templates
					- ![image.png](../assets/image_1676137724604_0.png)
			- Template Registration
			  collapsed:: true
				- TeTrIS: Template Transformer Networks for Image Segmentation
					- ![image.png](../assets/image_1676137748647_0.png)
					- ![image.png](../assets/image_1676137761687_0.png)
			- Structure-Guided Registration
				- ISTNs: Image-and-Spatial Transformer Networks
					- ![image.png](../assets/image_1676137792291_0.png)
					- ![image.png](../assets/image_1676137866601_0.png)
					- ![image.png](../assets/image_1676137875129_0.png)
					- ![image.png](../assets/image_1676137884096_0.png)
				- Atlas-ISTNs: Joint Segmentation, Registration and Atlas Construction
				  collapsed:: true
					- ![image.png](../assets/image_1676137916670_0.png)
					- ![image.png](../assets/image_1676137932553_0.png)
					- ![image.png](../assets/image_1676137950381_0.png)
					-
	- Week6
	- Week7
	- Week8
	- 考卷
		- 2021
		  collapsed:: true
			- logistic regression parameters 计算 includes raw pixels and a bias; e.g. 64 x 64 +1
			- Receptive Field 计算
			  collapsed:: true
				- 从外到内, 从上到下写.
				- 最外层的conv的kernel大小, 就是下一层conv需要生成的点数, 例如最外层是3
				- 根据下一层的kernel和stride, 画画图算一算找到下一层需要的像素数量, 例如第二层是2 stride2, 那么第二层的conv生成最外层的3所需的像素数量就是2*3=6
				- 再举个例子, 第三层是3 stride1, 因为stride是1, 所以除了前三个像素以外, 后面依次增加, 即3 + 5 = 8
				- 这样子算到最后一层即可
				- $RF_N=1,\  RF_{N-1}= (RF_N - 1)\times stride_n + k_n$
			- Convolution & pooling 计算
				- $o=[(i-k+2p)/s]+1$ - conv
				- $o = (i-1) \times s - 2p + k$ - transposed conv
			- Decision boundaries 计算
			- Accuracy, precision (positive predictive value, PPV), recall (sensitivity, hit rate, true positive rate, TPR), specificity (recall for neg, true negative rate, TNR), F-beta
				- F-\beta = (1 + \beta^2) * (Precision * Sensitivity) / (\beta^2 * Precision + Sensitivity)
			- SSD, SAD, CC, and NMI
			- Harr features: efficient using integral images within constant time; suitable for extracting structures like edges and corners, not general features, cannnot approximate any filters via convolution
			- Options for regularization in a neural network:
				- L1 & L2 penalty on the weights
				- Dropout
				- Stochastic Gradient Descent with small batch sizes
				- Data augmentation
			- YOLO
				- architecture and how it works
				  id:: 64120319-9f33-4806-90c3-6bd66a5caecc
				- how to avoid computational difficulties from 1000 classes
					- one could separate the object detection from the object classification. For example, one trains a generic object detector to detect regions of interest (as in R-CNN) and then a separate object classifier.
			- L1 & L2 loss
			- Perceptual loss #dj w7
				- Computes loss on the output 𝜃 of an intermediate layer 𝑙 of a pre-trained network:
				- 计算中间层的feature map的区别, 用的L2
			- Interpretability and Trustworthy ML #dj w8
				- interpret an existing ML model: saliency maps (突出重点信息)
					- Gradient (backpropagation) f>0
					- Deconvolution R>0
					- Guided backpropagation f & R >0
					- f is activated feature map; R is backpropogated value from l+1 to l
				- Adversarial attacks
					- Perturbation
						- 为了让扰动足够小, 我们限制了eta的大小为epsilon, $\eta = \epsilon · sign(\theta)$
						- 当权重theta乘上这个扰动的时候, 最终的扰动总和会是$\epsilon m n$
						- m表示某个theta的均值, n表示维度, 即像素数量, 可见这个扰动与n线性相关, 因此小eta也会带来大变化
						- This means that the change in activation given by the perturbation increases linearly with respect to 𝑛 (or the dimensionality). If 𝑛 is large, one can expect even a small perturbation capped at 𝜖 to produce a perturbation big enough to render the model susceptible to an adversarial attack.
					- Fast Gradient Sign Method
						- $\tilde{x} = x + \epsilon · sign(\nabla_{x}\mathcal{L}(\theta,x,y))$
						- 梯度下降减少梯度, 我们增加梯度, 对x增加梯度为正的部分, 每次增加epsilon
		- 1920
		  collapsed:: true
			- L1 and L2 regularisation
			- Mechanisms for fixing high variance
				- L1 and L2 regularisation
				- dropout
				- train with small batch
				- early stopping
				- reduce model complexity, flexibility (degree)
				- reduce features
				- data augmentation and more data
				- bagging
			- Mechanisms for fixing high bias
				- Increase model complexity, flexibility (degree)
				- add features
				- boosting
				- feature engineering
				- more training data
			- Partial Volume Effects #dj w4
				- Due to the coarse sampling, both tissue types (black and white) contribute to the intensity value of the generated image (right) due to the relatively large influence area of each pixel (gray square in left image)
			- encoder-decoder network design
				- tricks:
					- Encoder: (k, s, p), (5, 2, 2), (3,2,1) 可以减半(向上取整); (5,1,0) 减4, (3,1,0)减2
					- Decoder: (4,2,1) 可以增倍, (4,2,2)可以在增倍基础上减2, (4,2,3)基础上减3
			- Sketch the iterative process of intensity-based image registration. Include three main components of this process
				- Initialize the transformation parameters.
				- Apply the transformation model to the floating image using the current transformation parameters.
				- Calculate the similarity measure between the transformed floating image and the reference image.
				- Update the transformation parameters using the optimization strategy.
				- Repeat steps 2-4 until a termination criterion is met (e.g., the maximum number of iterations is reached or the change in similarity measure falls below a threshold).
				- Image transformation, Similarity measure using objective function, update the transformation parameters using optimisation strategy
			- spatial transformer network (STN)
				- It is a spatial transformer block which performs transformation on input data, make it invariant to translation, scale, rotation. With STN, network can learn to selectively apply spatial transformations to the input data, making it more robust to variations in scale, rotation, and other spatial factors.
			- Haar features
				- 使用integral images计算飞速, D-B-C+A
				- haar features可以scale为任意大小, 放在任意位置, 表示为白减黑的数值
			- RCNN, Fast-RCNN, Faster-RCNN, YOLO
			- Multi-label classificatin
				- This can be done by using a binary vector for each image, where each element of the vector corresponds to one of the possible classes, and a value of 1 indicates that the class is present in the image, while a value of 0 indicates that the class is absent.
				- Softmax and CE do not suit this case, as they assume there is only one correct class with sum of prob to be 1, while here we may have at most all ones. This may lead to unexpected results
				- Using sigmoid for each class and BCE to calculate the loss for each class and sum will be a better solution.
		- 1819
		  collapsed:: true
			- implementation of gradient descent
				- initialise weights, in training loop, calculate the predicted value, compute the loss, backpropogate the loss, compute the gradient for all samples and divided by number of samples, update weights by - alpha * gradient
			- implement stochastic gradient descent.
				- gradient calculated for each sample and update once per sample.
			- segmentation Overlapping measures
				- Jaccard Index (IoU, Jaccard Similarity Coefficient, JSC)
					- Intersection over Union
				- Dice’s Coefficient (Sørensen Index, Dice Similarity Coefficient, DSC) F1
					- $DSC = \frac{2|A ∩ B|}{|A|+|B|} = F1$
					- ![image.png](../assets/image_1678963717311_0.png)
					- Issue: Different shapes have identical DSC
				- volume similarity
					- ![image.png](../assets/image_1678963740289_0.png)
				- Hausdorff distance
					- ![image.png](../assets/image_1678963772894_0.png){:height 86, :width 287}
					- A上每个点找到B上离这个点的最短距离, 这些距离之中最长的就是我们求的这个距离
					- 相反也要找一遍, 取最大. 因为 b上凸起的部分, 这个时候才会发现距离贼大.
				- average surface distance
					- ![image.png](../assets/image_1678963937586_0.png)
					- A到B的最短距离的平均数
			- Ensemble learning methods
				- Decision tree, info gain, entropy, adaboost
			- Neural Networks
				- purpose of the 1 x 1 convolutions
					- Dimensionality reduction
					- Feature combination across channels
				- computational graph
		- 1718
		  collapsed:: true
			- image modality
				- NMI 适合多模态multimodal similarity 分析, 不适合
			- challenges that might effect the accuracy of an image segmentation algorithm
				- Partial volume effect
				- Intensity Inhomogeneities
				- Anisotropic Resolution
				- Imaging Artifacts
				- Limited Contrast
				- Morphological Variability
			- Leakage in image segmentation
				- 'leakage' refers to the phenomenon where the boundaries of segmented regions overlap or bleed into neighboring regions, leading to inaccurate segmentation results. Leakage can occur when the intensity or texture characteristics of neighboring regions are similar to the region of interest, leading to the misclassification of neighboring pixels as part of the segmented region.
			- Segmentation Algorithms
				- ▪ Intensity-based segmentation
				  ▪ e.g., thresholding
				- ▪ Region-based: region growing
					- Start from (user selected) seed point(s), and grow a region according to an intensity threshold
				- ▪ Graph-based: graph cuts
					- Segmentation based on max-flow/min-cut algorithm
				- ▪ Active contours: ▪ e.g., level sets
				- ▪ Atlas-based segmentation: multi-atlas label propagation
					- Use multiple atlas to register a new data and get multiple segmentations. Do majority voting for the final class.
				- ▪ Learning-based segmentation
				  ▪ e.g., random forests, convolutional neural networks
	- Tutorials
	  collapsed:: true
		- W2
			- MLI里面bias需要默认加上
			- conv操作默认不用flip, 但是可以自己注上去
		- W3
			- Haar features 适合提取edge, corner这种信息
			- Support 和 receptive field的计算
			- conv的parameters计算: kernels * (kernel size* input channels + bias)
		- W4
			- Ensemble methods的advantages 和disadv
			- Segmentation evaluation: 结合多个一起会比较有用一点
			- Pitfalls in segmentation evaluation
				- structure size
				- structure shape, spatial alignment 不能被DSC和IoU体现, 但可以从HD看出
				- hole: 有两层boundary, 都要看过, 内层离外边缘距离会远一点
				- annotation noise
				- resolution
			- challenges that might affect segmentation
				- ▪ noise,
				  ▪ partial volume effects,
				  ▪ intensity inhomogeneities, anisotropic resolution, ▪ imaging artifacts,
				  ▪ limited contrast,
				  ▪ morphological variability,
		- W5
			- (Dis)similarity measures
				- 如果各个组织的intensity间有一致的线性关系(例如contrast change), 此时可以使用 correlation coefficient; 如果线性关系不明确, 或完全就是两个modality, 那就得使用 NMI
				- Joint Histograms: 两个轴是两张图片的像素强度值, 散点是两张图每组对应像素的强度值分布, 如果整体是线性分布的, 突然出来的一块就是lesion之类的
			- Spatial Transformers
				- estimating the parameters of a **2D rigid** transformation for inputs of size 1 x 64 x 64. 最后要输出一个3维的(2个translation, 1个rotation) (3*2的affine matrix加上了scale和shear)
				- ![image.png](../assets/image_1679057312383_0.png)
		- W7
			- L1 L2 都要除以像素数量
			- Perceptual loss 用了几个filter, 就要除以几, 用的是L2
		- W8
			- saliency maps using three methods
			- Fast Gradient Sign Method: x + epsilon * gradient
	- 难点
		- Model complexity 一般指的就是parameter数量
		- Receptive Field 计算 #card
		  id:: 6411aa7f-80ee-4a0d-bdaf-e0e6b35cec02
		  collapsed:: true
			- 从外到内, 从上到下写.
			  id:: 6411aa8d-352f-4a94-898b-7e39bc7ac406
			- 最外层的conv的kernel大小, 就是下一层conv需要生成的点数, 例如最外层是3
			  id:: 6411aaa6-dea5-4525-95dd-1e4b719125fb
			- 根据下一层的kernel和stride, 画画图算一算找到下一层需要的像素数量, 例如第二层是2 stride2, 那么第二层的conv生成最外层的3所需的像素数量就是2*3=6
			  id:: 6411aad9-d4b0-45a1-91cb-dfbe745d79d4
			- 再举个例子, 第三层是3 stride1, 因为stride是1, 所以除了前三个像素以外, 后面依次增加, 即3 + 5 = 8
			  id:: 6411ab48-bb6f-4b24-959b-526e78050234
			- 这样子算到最后一层即可
			- •感受野计算公式：
			- 首先从最开始的layer开始，设定RF(receptive field) =1, SP(Stride Product)=1，然后当前层的感受野是 RF=RF + (kernel_size-1）* SP，然后SP也要根据当前的stride更新：SP=SP*stride
			- $RF_N=1,\  RF_{N-1}= (RF_N - 1)\times stride_n + k_n$
		- Convolution和Transposed Convolution的计算 #card
		  collapsed:: true
			- $o=[(i-k+2p)/s]+1$ - conv, 除法向下取整
			- $o = (i-1) \times s - 2p + k$ - transposed conv (输入元素间填充s-1, 四周填充k-p-1)
			- ```python
			  def conv_output(height, width, kernel_size, padding, stride, dilation): 
			  
			      height_out = int(np.floor(((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)) 
			  
			      width_out = int(np.floor(((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)) 
			  
			      print(height_out, width_out) 
			  
			      return height_out, width_out 
			  ```
		- AdaBoost
		- ensemble 的
		- W7 W8
		- W2
		  collapsed:: true
			- Gradient descent
				- ![image.png](../assets/image_1679062755582_0.png)
				- ![image.png](../assets/image_1679062818738_0.png)
			- Stochastic gradient descent
				- Update once per sample
			- One-vs-all: learn 𝑘-binary classifiers, 每个类都训练一个binary classifier
			-
		- W3
		  collapsed:: true
			- features: Distinctive, local, illumination invariant; intensity, gradient, SIFT, Harr
			- LBP: 九宫格第二个开始逆时针写vector, 比中间大为1
			- Haar
			- Ensemble classifiers
				- Error(x） =Bias^2 + Variance + Irreducible Error
				- Homogenous, Heterogenous: using same or different ML model classes
				- Decision Stump
					- a one-level decision tree that is axis-aligned, 在某个轴上大于或小于, 得到binary解
				- Bagging: bootstrap, 将训练集分几部分训练几个weak learner, aggregating, 汇总大家的答案
				- Boosting: build weak learners in serial, adaptively reweight training data prior to training each new weak learner in order to give a higher weight to previously misclassified examples, training the next learner on the mistakes of the previous one
					- ![image.png](../assets/image_1679066430042_0.png)
					- Adaboost: equal weighted samples, 分类, upweight misclassified, use loss to weight this classifier, 重复t次, 使用weighted classifiers 得到最后结果
				- Decision tree
					- ![image.png](../assets/image_1679066938090_0.png)
					- ![image.png](../assets/image_1679066945670_0.png)
					- Gini index: how mixed the classes are, 0 is best
						- ![image.png](../assets/image_1679067051183_0.png)
						- ![image.png](../assets/image_1679067056578_0.png)
			- Neural networks
				- multilayer perceptron 的话 会需要 neuron数量个bias
				- ![image.png](../assets/image_1679068069735_0.png)
				- ![IMG_1174.jpeg](../assets/IMG_1174_1679070695462_0.jpeg)
				- sigmoid': 0-0.25; tanh': 0-1
				- Stochastic gradient descent
					- ![image.png](../assets/image_1679070652989_0.png)
				- Residual connection
					- H(x)=F(x)+x
			-
		- W4
		  collapsed:: true
			- Challenges for Image Segmentation
			- Performance Measures
				- accuracy, precision, recall, specificity, F1 Score, Jaccard Index, Dice Similarity Coefficient (DSC), volume similarity,
			- Segmentation Algorithms & Techniques
				-
			- Fully Convolutional networks
			- Boundary Effects in transpose convolution
				- padding with 0 or nearest neighbour may give strange boundaries
			- Use only layers with small kernels to deal with deep issues
				- 2 3x3 instead of 1 5x5
			- Multi-Scale Processing
				- Two pathways, one using normal resolution, another with high resolution, concat.
				- This adds additional spatial info which enhances network's localisation capability
		- W5
		  collapsed:: true
			- ![image.png](../assets/image_1679075131522_0.png)
			- 注意rigid是2+1=3个参数, affine是6个
			- Free-Form Deformations: non-linear
			- Atlas construction
				- Iteration:
				- 1) Rigid
				- 2) Affine
				- 3) Nonrigid
			- Intensity-based Registration
				- Estimation of transformation parameters is driven by the appearance of the images
				- 找到让intensity relation 差异最小的那个T
				- (Dis)similarity Measures D
					- SSD, SAD (identity relationship, CTCT) mono-modal
					- Correlation coefficient(CC, linear relationship, MRMR)mono-modal
					- multi-modal have statistical relationship, CTMR, multi-modal
						- $p(i,j)=\frac{h(i,j)}{N}$ : 两张图对应像素点的强度为i和j的概率
						- ![image.png](../assets/image_1679084168261_0.png)
						- 熵越小越好, joint entropy也可以作为D
						- ![image.png](../assets/image_1679084227247_0.png)
						- mutual info, 表达了一张图被另外一张图的描述性, 越大越好, 用作D时取负
						- ![image.png](../assets/image_1679084330081_0.png)
						- normalised mutual info, 与overlap 无关, 同样越大越好, 用作D时取负
				- 因为要在overlap的部分衡量, 如果刚好只有背景重合也会有很好的成绩, 所以要小心的init
				- ![image.png](../assets/image_1679084646766_0.png){:height 287, :width 654}
			- Registration with Neural Networks
			- STN
		- W6
		  collapsed:: true
			- K-means
				- handles anisotropic data and non-linearities poorly
			- Gaussian Mixture Model (GMM)
				- ![image.png](../assets/image_1679086313611_0.png){:height 107, :width 260}
			- Dimensionality reduction / data encoding
				- PCA
					- ![image.png](../assets/image_1679086465067_0.png)
					- PCA is a linear encoder/decoder
				- Autoencoders
					- Auto-Encoders are general encoder/decoder architectures with non-linearities
					- non trivial to:
					  • visualize data in the latent space of Autoencoders
					  • generate new samples: perturb latent code of training example and iterate
			- Generative Modelling
				- Model the process through which data is generated
				  Capture the structure in the data
		- W7 inverse problem
		  collapsed:: true
			- Inverse problems
				- Inpainting, Deblurring, Denoising, Super-resolution
			- Classical approach: Least square solution, 用y和A直接解出x
			-
			- General approach:
				- ![image.png](../assets/image_1679089373884_0.png){:height 47, :width 305}
				- A是已知的, 想要从y学变回x
				- R是根据prior写的一个regulariser, 用来限制比如说像素的大小, 分布等等, 这里是人为使用像素的大小, 后续也可以用神经网络来找一个合适的先验, 例如图片就应该是类似这样分布的
			- Deep Learning approach
				- Model agnostic (ignores forward model)
					- 直接在逆向过程中接神经网络去学习, 从y到x的直接映射
				- Decoupled (First learn, then reconstruct)
					- Deep proximal gradient: z^k作为中间隐变量, 表示梯度下降更新以后的x^, 新的x^通过神经网络从z^k中生成, 学习从一个不健全的z生成科学的x的过程, denoising
					- GANs: 从G生成的图片里找到最适合data的那个
				- Unrolled optimisation
					- Gradient descent networks, R正则项也是个可导的神经网络, 通过不断迭代来寻找合适的重构和denoising方式
			- Super-Resolution
				- Post-upsampling super-resolution: 直接端到端, 需要学习整个pipeline, 限制在了特定的上采样factor
				- Pre-upsampling super-resolution: 先使用传统方法upsample到特定大小, 再用nn refine, 可以使用各种upscaling factors和images sizes, 需要更高的计算力和内存
				- Progressive upsampling super-resolution: multi-stage, upsample to a higher resolution and refined at each stage, efficient, but difficult to train deep models
				- Iterative up-and-down sampling super-resolution: Alternate between upsampling and downsampling (back-projection) operations, 且上下采样stages们是相互连接的, superior performance by allowing error feedback, easier to train with deep
			- Perceptual loss: 事实和估计都用同一个网络特征提取后的map进行L2 norm, 用了几个网络就把这几个结果加起来取平均
			- GAN:
				- ![image.png](../assets/image_1679091710772_0.png)
				- ![image.png](../assets/image_1679092208935_0.png)
			- Image Reconstruction
				- X-ray computed tomography (CT)
					- Sinogram to CT image
				- Magnetic Resonance (MR)
					- Slow acquisition, bad for moving objects
					- performed in k-space by sequentially traversing sampling trajectories. need to inverse to signal space, IFT
			-
		- W7 Object detection
		  collapsed:: true
			- classification(softmax) + localisation(L2)
			- 因为不可能sliding window全部遍历, 所以需要region proposal
			- R-CNN
				- ![image.png](../assets/image_1679093581412_0.png){:height 349, :width 654}
			- Spatial Pyramid Pooling (SPP-net)
				- ![image.png](../assets/image_1679095241578_0.png)
				- 在FM上region proposal, 只计算一次feature, 通过SPP把regions变成统一长度向量; SPP下无法更新
			- Fast R-CNN
				- 在图片里region proposal, 映射到FM中, 通过ROI pooling变成7x7, FC给分类和回归, 这里的分类用的也是FC和softmax
				- ![image.png](../assets/image_1679095599657_0.png)
			- Faster R-CNN
				- Insert Region Proposal Network (RPN) to predict proposals from features
				- 1. RPN classify object / not object
				  2. RPN regress box coordinates
				  3. Final classification score (object classes)
				  4. Final box coordinates
				- 用RPN来预测feature map上是否为object, 以及bounding box
			- YOLO
				- 分割图片成7x7个格子, 每个格子负责预测一定数量的bounding box, 有4个 coordinates, 1个confidence score, 以及1个类别数dim的分类vector
					- 7 x 7 x (2 x 5 + 20)
				- perform NMS and threshold 来选择好用的, increase the confidence for the best ones, decrease others
		- W8 trustworthy
			- Federated learning: train a ML model across decentralized clients with local data, without exchanging them
			- Federated learning: 分发模型给clients, 各自用自己数据计算模型更新, 发回owner来aggregate
				- ![image.png](../assets/image_1679133860277_0.png){:height 256, :width 237}
				- 也可以直接更新weights, server那也直接average weights
				- 挑战: non-iid, unbalanced, commu cost
			-
			- Homomorphic encryption which enables learning from encrypted data
			- Secure multi-party computing where processing is performed on encrypted data shares, split among them in a way that no single party can retrieve the entire data on their own. 共同计算一个平均分, 第一个人先加个随机数, 单向传播, 回到自己的时候可以计算平均分
			- Interpretability
				- social issue, debuging
				- Ablation test: How important is a data point or feature?
				- Fit functions (use first derivatives, sensitivity and saliency maps)
				- visualize activations generated by kernels
				- Mask out region in the input image and observe network output
				- DeconvNet: Chose activation at one layer, 取gradient中非负的数
				- Gradient (backprogagation): differentiate activation with respect to input pixels, 取gradient里activation激活的对应位置项
				- Guided backpropagation: gradient里面为负的也不取
				- ![image.png](../assets/image_1679136454035_0.png)
				- Inverting codes via deep image priors: 生成网络从z0生成x, loss由另一个特征提取网络对原图和生成图的结果差得到、
				- Learning the inverter from data: 生成网络从特征提取网络得到的特征开始生成x, 直接与原图计算loss
			- DeepDream和Inversion是非常相似的方法，它们试图将神经网络的工作内容和工作方式可视化。Deep Image的先验是不同的，它为图像指定了一个特定的先验（而不是使用L1 norm这样的先验）。DeepDream和Inversion可以使用不同的先验。
			- Adversarial Methods
				- ![image.png](../assets/image_1679137387448_0.png)
				- ![image.png](../assets/image_1679137396653_0.png)
				- ![image.png](../assets/image_1679137459790_0.png)
			-
	- 疑点:
		-
- ## Coursework
  collapsed:: true
	- Age Regression from Brain MRI
	  ▪ mini-project on a real-world medical imaging task
	  ▪ implement two different machine learning approaches ▪ work in groups of two
	- Start: Friday, February 17 (week 6)
	- End: Thursday, March 2 (week 8)
	-
- ## Info
  collapsed:: true
	- 8:2
	- ▪ Fridays, 14:00, LT308: Q&A, invited talks, quizzes
	  ▪ Fridays, 15:00, LT202/206/210 : programming tutorials
- ## Syllabus
  collapsed:: true
	- Introduction to machine learning for imaging
	- Image classification
	- Image segmentation
	- Object detection & localisation
	- Image registration
	- Generative models and representation learning
	- Application to real-world problems
- ## Links
  collapsed:: true
	- [Scientia](https://scientia.doc.ic.ac.uk/2223/modules/70014/materials)
	- [Panopto](https://imperial.cloud.panopto.eu/Panopto/Pages/Sessions/List.aspx#folderID=%229755113e-85d4-4a84-afcc-aedd01525333%22)
	- [Spatial Transformer Networks Tutorial — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
	- [GitHub - ecekayan/mli-cw: Machine Learning for Imaging CW](https://github.com/ecekayan/mli-cw)
	- [GitHub - marektopolewski/ic-mli-cw](https://github.com/marektopolewski/ic-mli-cw)
	- [GitHub - Cy-r0/mli_cw2: Second coursework of Machine Learning for Imaging](https://github.com/Cy-r0/mli_cw2)
	-
