# Features
## Local Features
### 1. Point Feature Histogram PFH
[reference link](https://blog.csdn.net/xinxiangwangzhi_/article/details/90023207)  

特点：
+ 复杂度 nk^2

##### 转换成直方图
- 四个特征，每个特征平均分成b份（pcl中为5），在实际应用时，距离特征可以不用，此时一共有b^3种情况，即b=5时，拥有125种情况，即125个bin的直方图
- 这样有一个问题：对于点云特别是稀疏点云来说，很多区间存在0值，即直方图上存在冗余空间

### 2. Fast Point Feature Histogram FPFH  
[reference link](https://blog.csdn.net/xinxiangwangzhi_/article/details/90023207)    

特点：
+ 尺度和姿态的不变性
+ 复杂度 nk

##### 转换成直方图
通过分解三元组（三个角特征）简化了合成的直方图，即简单地创建b个相关的的特征直方图，每个特征维数（dimension）对应一个直方图（bin），并将它们连接在一起。pcl默认FPFH的b=11，3*11=33，也是FPFHSignature33。    


### 直方图相似度计算方法

## Globle Features
### 1. Viewpoint Feature Histogram VFH
VFH特征包含两部分：
1. viewpoint direction component
计算点云centroid Pc；计算向量观察点原点到Pc 和 点云其他所有点的normal向量之间的夹角，统计直方图128bin
为了保持尺度不变性
2. extened FPFH component
类似pfh计算三个角度，区别是只计算所有点与重心点的，统计直方图，每个角度分成45个bin，共
3. 直方图拼接
128+45*3 = 263 
