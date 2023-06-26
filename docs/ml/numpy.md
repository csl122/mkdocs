# NumPy

tags:: python, package,

-
- 实用方法
	- ```python
	  numpy 与 torch 类似, 看括号的个数可以知道维度
	  
	  a = np.array(999)
	  a.ndim
	  -> 0
	  
	  # shape 可以得知形状
	  b = np.array([[1],[1]])
	  b.shape
	  -> (2, 1)
	  
	  # size 可以得知总共元素数量
	  b.size
	  -> 2
	  
	  # dtype 可以得知数据类型
	  b.dtype
	  -> int64
	  
	  # reshape 可以更改形状
	  b.reshape(1,2)
	  -> [[1 1]]
	  
	  # full 指定填充值
	  nines = np.full((2,3,4), 9)
	  
	  # 利用dtype定义数据类型
	  arr = np.array([1,2,3], dtype=np.int64)
	  
	  # where 函数 筛选条件, 从哪里筛选, 不符合的显示
	  np.where(mask, student_marks, np.nan)
	  
	  # argwhere 得到索引
	  np.argwhere(mask)
	  
	  # 轴向, axis 在max等函数中的理解
	  max函数为例, axis=0, 就是在0轴变化方向压缩成只有一个值
	  二维数组就会只剩下一行, 但是列保持不变
	  三维数组就会只剩下一个二维数组, 但是每个二维数组的行和列不变
	  
	  # copy方法创建一个独立内存的拷贝, 不会互相影响
	  c = b.copy()
	  
	  # unique 去重
	  d = np.unique(c)
	  
	  # concatenate 尾首拼接
	  np.concatenate((arr1, arr2))
	  
	  # insert 插入指定位置
	  np.insert(arr2, 1, arr1)
	  
	  # delete 删除元素
	  np.delete(arr3, 0) 会返回被删除元素
	  
	  ```
- Tutorial
	- [Introduction to NumPy and Matplotlib > Chapter 1: Introduction | Python Programming (70053 Autumn Term 2022/2023) | Department of Computing | Imperial College London](https://python.pages.doc.ic.ac.uk/2022/lessons/numpy/01-intro/)
	- [Python Numpy Tutorial (with Jupyter and Colab)](https://cs231n.github.io/python-numpy-tutorial/)
	-
-
