# Feature Pyramid Networks

## 网络结构
#### 1. Bottom up
作者使用resnet作为backbone。将输出相同尺寸的feature map层称为一个**stage**，对于resnet，作者选取每个stage的最后一个residual block的激活输出作为预测map；一共有四个，对应stride{4,8,16,32}相对于原图，不使用stride=2的feature map原因是太大了。
#### 2. Top down 
这里上采样使用了**最邻近点插值**；bottom up输出的feature map经过1x1卷积进行深度通道改变（减少），已匹配上采样的通道数。  
融合后的feature经过3x3卷积，防止混叠，同时固定所有的融合feature的深度为一样的值=256
![alt](imgs/fpn1.webp)

## 结合应用
#### 1.RPN
* 如下图，分别生成了5个feature map，每个map有对应不同的anchor设置，分别为
    - 面积{$32^2, 64^2，128^2， 256^2，512^2$}（在原图大小下的）
    - 比例{0.5,1，2}
同样的浅层的feature，使用的size越小，为了检测小的物体 

* GT的判断方法同faster-rcnn

* 所有的feature都共享一套之后的处理参数，即3x3卷积+两个平行1x1卷积（分类+定位）；作者尝试使用不同的head，提升不明显
![alt](imgs/fpn.png)
#### 2.Fast-Rcnn / Faster-Rcnn
- fast rcnn是需要将proposal映射到feature map上，同样的faster-rcnn在求出proposals后，也要映射回原图像。接下来，RoIs映射到feature maps上时，由于有多层feature map,需要判断哪个ＲＯＩｓ对应哪个ｍａｐ：
$$k=\lfloor k_0+log_2(\sqrt{wh}/224) \rfloor$$
    wh：ａｎｃｈｏｒ对应原图的尺寸
    224是因为网络在imgaenet上预训练的；
    $k_0=4$表示一个大小为224*224的anchor应该被映射到的feature map,根据ResNet-based faster rcnn结构（Ｃ４层），这里设置为４；也就是说越小的anchor越应该映射到分辨率高的feature map上，例如wh=112*112，则k=3

- 所有的ＲＯＩｓ共享一个head,同ＲＰＮ的应用方式  

（个人想法：为什么faster-rcnn已经在每个feature map上计算出了proposals，不是应该知道对应的feature map么？
可能由于proposal需要进行regression，之后的size可能发生变化，所以最好从新计算对应的feature map）
![alt](imgs/fpn.jpeg)

![alt](imgs/fpn2)

## referencec
[paper](https://zpascal.net/cvpr2017/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)