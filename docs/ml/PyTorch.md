Pytorch

tags::  python, package, ML
alias:: Torch

- Installation #安装
	- ```
	  Linux
	  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	  pip3 install torch torchvision torchaudio
	  
	  Mac
	  conda install pytorch torchvision torchaudio -c pytorch
	  pip3 install torch torchvision torchaudio
	  ```
-
- Tensor
  collapsed:: true
	- ```python
	  # 数据类型的种类
	  torch.float64(torch.double), 
	  
	  **torch.float32(torch.float)**, 
	  
	  torch.float16, 
	  
	  torch.int64(torch.long), 
	  
	  torch.int32(torch.int), 
	  
	  torch.int16, 
	  
	  torch.int8, 
	  
	  torch.uint8, 
	  
	  torch.bool
	  
	  #数据类型的判断
	  x = torch.tensor(2.0);print(x,x.dtype)
	  tensor(2.) torch.float32
	  
	  #数据类型的指定
	  i = torch.tensor(1,dtype = torch.int32)
	  
	  #特定数据类型的构造器
	  i = torch.IntTensor(1);print(i,i.dtype)
	  x = torch.Tensor(np.array(2.0));print(x,x.dtype) #等价于torch.FloatTensor
	  b = torch.BoolTensor(np.array([1,0,2,0])); print(b,b.dtype)
	  
	  #数据类型的转换
	  i = torch.tensor(1); print(i,i.dtype)
	  x = i.float(); print(x,x.dtype) #调用 float方法转换成浮点类型
	  y = i.type(torch.float); print(y,y.dtype) #使用type函数转换成浮点类型
	  z = i.type_as(x);print(z,z.dtype) #使用type_as方法转换成某个Tensor相同类型
	  
	  
	  # 张量的维度
	  标量为0维张量，向量为1维张量，矩阵为2维张量。 有几层中括号，就是多少维的张量。
	  
	  # 0维张量
	  scalar = torch.tensor(True)
	  print(scalar)
	  print(scalar.dim())  # 标量，0维张量
	  
	  # 1维张量
	  vector = torch.tensor([1.0,2.0,3.0,4.0]) #向量，1维张量
	  print(vector)
	  print(vector.dim())
	  
	  # 2维张量
	  matrix = torch.tensor([[1.0,2.0],[3.0,4.0]]) #矩阵, 2维张量
	  print(matrix)
	  print(matrix.dim())
	  
	  # 4维张量
	  tensor4 = torch.tensor([[[[1.0,1.0],[2.0,2.0]],[[3.0,3.0],[4.0,4.0]]],
	                          [[[5.0,5.0],[6.0,6.0]],[[7.0,7.0],[8.0,8.0]]]])  # 4维张量
	  print(tensor4)
	  print(tensor4.dim())
	  
	  
	  # 张量的尺寸形状
	  shape属性和size()方法可以用来查看, view()和reshape()方法可以用来改变
	  
	  # 0维张量尺寸
	  torch.Size([])
	  
	  # 1维张量尺寸
	  vector = torch.tensor([1.0,2.0,3.0,4.0])
	  print(vector.size())
	  print(vector.shape)
	  -> torch.Size([4])
	  
	  # 使用view可以改变张量尺寸
	  vector = torch.arange(0,12)
	  print(vector)
	  print(vector.shape)
	  
	  matrix34 = vector.view(3,4)
	  print(matrix34)
	  print(matrix34.shape)
	  
	  matrix43 = vector.view(4,-1) #-1表示该位置长度由程序自动推断
	  print(matrix43)
	  print(matrix43.shape)
	  
	  
	  # 有些操作会让张量存储结构扭曲，直接使用view会失败，可以用reshape方法
	  # [python - What does .contiguous() do in PyTorch? - Stack Overflow](https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch)
	  # 即index的摆放方法和内存中的摆放方式不一致了
	  matrix26 = torch.arange(0,12).view(2,6)
	  print(matrix26)
	  print(matrix26.shape)
	  
	  # 转置操作让张量存储结构扭曲
	  matrix62 = matrix26.t()
	  print(matrix62.is_contiguous())
	  
	  
	  # 直接使用view方法会失败，可以使用reshape方法
	  #matrix34 = matrix62.view(3,4) #error!
	  matrix34 = matrix62.reshape(3,4) #等价于matrix34 = matrix62.contiguous().view(3,4)
	  print(matrix34)
	  
	  
	  # 张量和numpy之间的互相转换和联系
	  numpy()方法和from_numpy(arr)可以互相转换两种类型, 但是共享内存, 改一个两个都变
	  arr = np.zeros(3)
	  tensor = torch.from_numpy(arr)
	  print("before add 1:")
	  
	  print("\nafter add 1:")
	  np.add(arr,1, out = arr) #给 arr增加1，tensor也随之改变
	  
	  tensor = torch.zeros(3)
	  arr = tensor.numpy()
	  print("before add 1:")
	  
	  print("\nafter add 1:")
	  
	  #使用带下划线的方法表示计算结果会返回给调用 张量
	  tensor.add_(1) #给 tensor增加1，arr也随之改变 
	  #或： torch.add(tensor,1,out = tensor)
	  
	  #clone()方法可以拷贝而非链接
	  tensor = torch.zeros(3)
	  
	  #使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
	  arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
	  
	  # item方法和tolist方法可以将张量转换成Python数值和数值列表
	  scalar = torch.tensor(1.0)
	  s = scalar.item()
	  print(s)
	  print(type(s))
	   <class 'float'>
	  
	  tensor = torch.rand(2,2)
	  t = tensor.tolist()
	  print(t)
	  print(type(t))
	   <class 'list'>
	  
	  ```
- 张量结构操作
  collapsed:: true
	- ```python
	  tensor, arange, linspace, zeros, ones, zeros_like, fill_, full, rand, normal, randn, randperm, eye, diag
	  import numpy as np
	  import torch 
	  
	  # 默认创建方式
	  a = torch.tensor([1,2,3],dtype = torch.float)
	  tensor([1., 2., 3.])
	  
	  b = torch.arange(1,10,step = 2)
	  tensor([1, 3, 5, 7, 9])
	  
	  c = torch.linspace(0.0,2*3.14,10)
	  tensor([0.0000, 0.6978, 1.3956, 2.0933, 2.7911, 3.4889, 4.1867, 4.8844, 5.5822,
	          6.2800])
	          
	  d = torch.zeros((3,3))
	  tensor([[0., 0., 0.],
	          [0., 0., 0.],
	          [0., 0., 0.]])
	          
	  a = torch.ones((3,3),dtype = torch.int)
	  b = torch.zeros_like(a,dtype = torch.float)
	  tensor([[1, 1, 1],
	          [1, 1, 1],
	          [1, 1, 1]], dtype=torch.int32)
	  tensor([[0., 0., 0.],
	          [0., 0., 0.],
	          [0., 0., 0.]])
	          
	  torch.fill_(b,5)
	  tensor([[5., 5., 5.],
	          [5., 5., 5.],
	          [5., 5., 5.]])
	          
	  x = torch.full((3, 5), 3.0)
	  tensor([[3., 3., 3., 3., 3.],
	          [3., 3., 3., 3., 3.],
	          [3., 3., 3., 3., 3.]])
	  
	  #均匀随机分布 uniform distribution
	  torch.manual_seed(0)
	  minval,maxval = 0,10
	  a = minval + (maxval-minval)*torch.rand([5])
	  tensor([4.9626, 7.6822, 0.8848, 1.3203, 3.0742])
	  
	  #正态分布随机 normal distribution
	  b = torch.normal(mean = torch.zeros(3,3), std = torch.ones(3,3))
	  tensor([[ 0.5507,  0.2704,  0.6472],
	          [ 0.2490, -0.3354,  0.4564],
	          [-0.6255,  0.4539, -1.3740]])
	          
	  #正态分布随机
	  mean,std = 2,5
	  c = std*torch.randn((3,3))+mean
	  tensor([[16.2371, -1.6612,  3.9163],
	          [ 7.4999,  1.5616,  4.0768],
	          [ 5.2128, -8.9407,  6.4601]])
	  
	  #整数随机排列
	  d = torch.randperm(20)
	  tensor([ 3, 17,  9, 19,  1, 18,  4, 13, 15, 12,  0, 16,  7, 11,  2,  5,  8, 10,
	           6, 14])
	  
	  #特殊矩阵
	  I = torch.eye(3,3) #单位矩阵
	  print(I)
	  t = torch.diag(torch.tensor([1,2,3])) #对角矩阵
	  print(t)
	  tensor([[1., 0., 0.],
	          [0., 1., 0.],
	          [0., 0., 1.]])
	  tensor([[1, 0, 0],
	          [0, 2, 0],
	          [0, 0, 3]])
	  
	  
	  
	  
	  ```
- 张量索引和切片
  collapsed:: true
	- ```python
	  # 索引和切片
	  #第0行
	  print(t[0])
	  #倒数第一行
	  print(t[-1])
	  #第1行第3列
	  print(t[1,3])
	  print(t[1][3])
	  #第1行至第3行 [1,4)
	  print(t[1:4,:])
	  #第1行至最后一行，第0列到最后一列每隔两列取一列
	  print(t[1:4,:4:2])
	  #可以使用索引和切片修改部分元素
	  x = torch.Tensor([[1,2],[3,4]])
	  x.data[1,:] = torch.tensor([0.0,0.0])
	  
	  a = torch.arange(27).view(3,3,3)
	  tensor([[[ 0,  1,  2],
	           [ 3,  4,  5],
	           [ 6,  7,  8]],
	  
	          [[ 9, 10, 11],
	           [12, 13, 14],
	           [15, 16, 17]],
	  
	          [[18, 19, 20],
	           [21, 22, 23],
	           [24, 25, 26]]])
	  #省略号可以表示多个冒号, 所有块, 所有列的第1列
	  print(a[:,:,1]) == print(a[...,1])
	  tensor([[ 1,  4,  7],
	          [10, 13, 16],
	          [19, 22, 25]])
	  
	  
	  对于不规则的切片提取,可以使用torch.index_select, torch.masked_select, torch.take, torch.gather
	  考虑班级成绩册的例子，有4个班级，每个班级5个学生，每个学生7门科目成绩。可以用一个4×5×7的张量来表示。
	  tensor([[[55, 95,  3, 18, 37, 30, 93],
	           [17, 26, 15,  3, 20, 92, 72],
	           [74, 52, 24, 58,  3, 13, 24],
	           [81, 79, 27, 48, 81, 99, 69],
	           [56, 83, 20, 59, 11, 15, 24]],
	  
	          [[72, 70, 20, 65, 77, 43, 51],
	           [61, 81, 98, 11, 31, 69, 91],
	           [93, 94, 59,  6, 54, 18,  3],
	           [94, 88,  0, 59, 41, 41, 27],
	           [69, 20, 68, 75, 85, 68,  0]],
	  
	          [[17, 74, 60, 10, 21, 97, 83],
	           [28, 37,  2, 49, 12, 11, 47],
	           [57, 29, 79, 19, 95, 84,  7],
	           [37, 52, 57, 61, 69, 52, 25],
	           [73,  2, 20, 37, 25, 32,  9]],
	  
	          [[39, 60, 17, 47, 85, 44, 51],
	           [45, 60, 81, 97, 81, 97, 46],
	           [ 5, 26, 84, 49, 25, 11,  3],
	           [ 7, 39, 77, 77,  1, 81, 10],
	           [39, 29, 40, 40,  5,  6, 42]]], dtype=torch.int32)
	  #抽取每个班级第0个学生，第2个学生，第4个学生的全部成绩; 选学生, 所以取dim 1
	  torch.index_select(scores,dim = 1,index = torch.tensor([0,2,4]))
	  tensor([[[55, 95,  3, 18, 37, 30, 93],
	           [74, 52, 24, 58,  3, 13, 24],
	           [56, 83, 20, 59, 11, 15, 24]],
	  
	          [[72, 70, 20, 65, 77, 43, 51],
	           [93, 94, 59,  6, 54, 18,  3],
	           [69, 20, 68, 75, 85, 68,  0]],
	  
	          [[17, 74, 60, 10, 21, 97, 83],
	           [57, 29, 79, 19, 95, 84,  7],
	           [73,  2, 20, 37, 25, 32,  9]],
	  
	          [[39, 60, 17, 47, 85, 44, 51],
	           [ 5, 26, 84, 49, 25, 11,  3],
	           [39, 29, 40, 40,  5,  6, 42]]], dtype=torch.int32)
	  
	  #抽取每个班级第0个学生，第2个学生，第4个学生的第1门课程，第3门课程，第6门课程成绩
	  q = torch.index_select(torch.index_select(scores,dim = 1,index = torch.tensor([0,2,4]))
	                     ,dim=2,index = torch.tensor([1,3,6]))
	  tensor([[[95, 18, 93],
	           [52, 58, 24],
	           [83, 59, 24]],
	  
	          [[70, 65, 51],
	           [94,  6,  3],
	           [20, 75,  0]],
	  
	          [[74, 10, 83],
	           [29, 19,  7],
	           [ 2, 37,  9]],
	  
	          [[60, 47, 51],
	           [26, 49,  3],
	           [29, 40, 42]]], dtype=torch.int32)
	  
	  #抽取第0个班级第0个学生的第0门课程，第2个班级的第3个学生的第1门课程，第3个班级的第4个学生第6门课程成绩
	  #take将输入看成一维数组，输出和index同形状
	  s = torch.take(scores,torch.tensor([0*5*7+0,2*5*7+3*7+1,3*5*7+4*7+6]))
	  tensor([55, 52, 42], dtype=torch.int32)
	  
	  #抽取分数大于等于80分的分数（布尔索引）
	  #结果是1维张量
	  g = torch.masked_select(scores,scores>=80)
	  tensor([95, 93, 92, 81, 81, 99, 83, 81, 98, 91, 93, 94, 94, 88, 85, 97, 83, 95,
	          84, 85, 81, 97, 81, 97, 84, 81], dtype=torch.int32)
	  
	  
	  以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。
	  
	  如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where,torch.index_fill 和 torch.masked_fill
	  
	  torch.where可以理解为if的张量版本。
	  
	  torch.index_fill的选取元素逻辑和torch.index_select相同。
	  
	  torch.masked_fill的选取元素逻辑和torch.masked_select相同。
	  #如果分数大于60分，赋值成1，否则赋值成0
	  ifpass = torch.where(scores>60,torch.tensor(1),torch.tensor(0))
	  
	  #将每个班级第0个学生，第2个学生，第4个学生的全部成绩赋值成满分
	  torch.index_fill(scores,dim = 1,index = torch.tensor([0,2,4]),value = 100)
	  #等价于 scores.index_fill(dim = 1,index = torch.tensor([0,2,4]),value = 100)
	  
	  #将分数小于60分的分数赋值成60分
	  b = torch.masked_fill(scores,scores<60,60)
	  #等价于b = scores.masked_fill(scores<60,60)
	  
	  
	  
	  
	  
	  
	  
	  
	  ```
- 张量维度变换
  collapsed:: true
	- ```python
	  维度变换相关函数主要有 torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose
	  
	  torch.reshape 可以改变张量的形状。
	  
	  torch.squeeze 可以减少维度。
	  
	  torch.unsqueeze 可以增加维度。
	  
	  torch.transpose/torch.permute 可以交换维度。
	  
	  minval,maxval = 0,255
	  a = (minval + (maxval-minval)*torch.rand([1,3,3,2])).int()
	  torch.Size([1, 3, 3, 2])
	  tensor([[[[126, 195],
	            [ 22,  33],
	            [ 78, 161]],
	  
	           [[124, 228],
	            [116, 161],
	            [ 88, 102]],
	  
	           [[  5,  43],
	            [ 74, 132],
	            [177, 204]]]], dtype=torch.int32)
	  
	  
	  # 改成 （3,6）形状的张量
	  b = a.view([3,6]) #torch.reshape(a,[3,6])
	  tensor([[126, 195,  22,  33,  78, 161],
	          [124, 228, 116, 161,  88, 102],
	          [  5,  43,  74, 132, 177, 204]], dtype=torch.int32)
	          
	  如果张量在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度。
	  a = torch.tensor([[1.0,2.0]])
	  s = torch.squeeze(a)
	  tensor([[1., 2.]])
	  tensor([1., 2.])
	  torch.Size([1, 2])
	  torch.Size([2])
	  
	  d = torch.unsqueeze(s,axis=0)  
	  tensor([1., 2.])
	  tensor([[1., 2.]])
	  torch.Size([2])
	  torch.Size([1, 2])
	  
	  
	  torch.transpose可以交换张量的维度，torch.transpose常用于图片存储格式的变换上。
	  
	  如果是二维的矩阵，通常会调用矩阵的转置方法 matrix.t()，等价于 torch.transpose(matrix,0,1)。
	  minval=0
	  maxval=255
	  # Batch,Height,Width,Channel
	  data = torch.floor(minval + (maxval-minval)*torch.rand([100,256,256,4])).int()
	  print(data.shape)
	  
	  # 转换成 Pytorch默认的图片格式 Batch,Channel,Height,Width 
	  # 需要交换两次
	  data_t = torch.transpose(torch.transpose(data,1,2),1,3)
	  print(data_t.shape)
	  
	  
	  data_p = torch.permute(data,[0,3,1,2]) #对维度的顺序做重新编排
	  data_p.shape 
	  
	  
	  
	  ```
- 张量合并分割
  collapsed:: true
	- ```python
	  可以用torch.cat方法和torch.stack方法将多个张量合并，可以用torch.split方法把一个张量分割成多个张量。
	  
	  torch.cat和torch.stack有略微的区别，torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度。
	  
	  a = torch.tensor([[1.0,2.0],[3.0,4.0]])
	  b = torch.tensor([[5.0,6.0],[7.0,8.0]])
	  c = torch.tensor([[9.0,10.0],[11.0,12.0]])
	  
	  abc_cat = torch.cat([a,b,c],dim = 0)
	  torch.Size([6, 2])
	  tensor([[ 1.,  2.],
	          [ 3.,  4.],
	          [ 5.,  6.],
	          [ 7.,  8.],
	          [ 9., 10.],
	          [11., 12.]])
	          
	  abc_stack = torch.stack([a,b,c],axis = 0) #torch中dim和axis参数名可以混用
	  print(abc_stack.shape)
	  print(abc_stack)
	  torch.Size([3, 2, 2])
	  tensor([[[ 1.,  2.],
	           [ 3.,  4.]],
	  
	          [[ 5.,  6.],
	           [ 7.,  8.]],
	  
	          [[ 9., 10.],
	           [11., 12.]]])
	  
	  torch.cat([a,b,c],axis = 1)
	  tensor([[ 1.,  2.,  5.,  6.,  9., 10.],
	          [ 3.,  4.,  7.,  8., 11., 12.]])
	  
	  torch.stack([a,b,c],axis = 1)
	  tensor([[[ 1.,  2.],
	           [ 5.,  6.],
	           [ 9., 10.]],
	  
	          [[ 3.,  4.],
	           [ 7.,  8.],
	           [11., 12.]]])
	  
	  torch.split是torch.cat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。
	  print(abc_cat)
	  a,b,c = torch.split(abc_cat,split_size_or_sections = 2,dim = 0) #每份2个进行分割
	  print(a)
	  print(b)
	  print(c)
	  tensor([[ 1.,  2.],
	          [ 3.,  4.],
	          [ 5.,  6.],
	          [ 7.,  8.],
	          [ 9., 10.],
	          [11., 12.]])
	  tensor([[1., 2.],
	          [3., 4.]])
	  tensor([[5., 6.],
	          [7., 8.]])
	  tensor([[ 9., 10.],
	          [11., 12.]])
	  
	  
	  
	  
	  
	  
	  ```
-
- 模型 #model #GitHub #ML
	- [vision/torchvision/models at main · pytorch/vision · GitHub](https://github.com/pytorch/vision/tree/main/torchvision/models)
	- [AI Models - Computer Vision, Conversational AI, and More | NVIDIA NGC](https://catalog.ngc.nvidia.com/models)
	- [For Researchers | PyTorch](https://pytorch.org/hub/research-models)
	- https://modelzoo.co/framework/pytorch
	- [vision/torchvision/models at main · pytorch/vision · GitHub](https://github.com/pytorch/vision/tree/main/torchvision/models)
	- [GitHub - Cadene/pretrained-models.pytorch: Pretrained ConvNets for pytorch: NASNet, ResNeXt, ResNet, InceptionV4, InceptionResnetV2, Xception, DPN, etc.](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
	- https://modelzoo.co/model/pretrained-modelspytorch
	-
-
- PyTorch Container #Docker #Container #Linux #unix
	- [PyTorch Release Notes :: NVIDIA Deep Learning Frameworks Documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-01.html#rel-23-01)
	- [PyTorch | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
	-
- PyTorch Documentation
	- [ResNet — Torchvision main documentation](https://pytorch.org/vision/stable/models/resnet.html)
	- [Tensors — PyTorch Tutorials 1.13.1+cu117 documentation](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
	-
-
