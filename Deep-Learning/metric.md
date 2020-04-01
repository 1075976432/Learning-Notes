# 分割指标
![alt](imgs/segmetric.png)
假设如下：共有k+1个类（从L0到Lk，其中包含一个空类或背景），pij表示本属于类i但被预测为类j的像素数量。即，pii表示真正的数量，而pij  pji则分别被解释为假正和假负，尽管两者都是假正与假负之和。
- Pixel Accuracy(PA，像素精度)：这是最简单的度量，为标记正确的像素占总像素的比例
![alt](imgs/segmetric1.png)
- Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均
![alt](imgs/segmetric2.png)
- Mean Intersection over Union(MIoU，均交并比)：为语义分割的标准度量。其计算两个集合的交集和并集之比，在语义分割的问题中，这两个集合为真实值（ground truth）和预测值（predicted segmentation）。这个比例可以变形为正真数（intersection）比上真正、假负、假正（并集）之和。在每个类上计算IoU，之后平均
![alt](imgs/segmetric3.png)
- frequency weighted IU
![alt](imgs/segmetric4.png)