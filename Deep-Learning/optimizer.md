## 梯度下降法　Gradient Descent
假设要学习训练的模型参数为W，代价函数为J(W)，则代价函数关于模型参数的偏导数即相关梯度为ΔJ(W)，学习率为η，则使用梯度下降法更新参数为：
$$W_{t+1}=W_t-\eta_t \Delta J(W_t)$$
### 1. Batch Gradient Descent
每次更新时，**所有的数据**都参与计算梯度

优缺点：
- 每步确保为真实梯度方向　（参考了所有数据）
- 训练速度慢，计算量大
- 容易陷入局部最优解 
- 添加新的样本，需要重新计算所有样本的梯度

### 2. Stochastic Gradient Descent
每次更新时，只使用**一个样本**计算梯度
而对于引入噪声，大量的理论和实践工作证明，只要噪声不是特别大，SGD都能很好地收敛。
优点：
- 训练速度很快，
- 计算量小。
- 可以随时添加新的样本
缺点：
- 引入噪声，更新抖动，使得权值更新的方向不一定正确。
- 局部最优解的问题。

### ３. Mini Batch Gradient descent
每次更新时，使用**部分样本**计算梯度
优点：
- 减少计算梯度的方差，有一个稳定的收敛
- 加速学习过程

## 动量优化法
动量优化方法是在梯度下降法的基础上进行的改变，具有加速梯度下降的作用。增加动量，使优化器更容易走出局部最优解
### 1. Momentum
- 有助于在某一方向梯度很大的情况下，加速收敛；
- 抑制梯度方向振动。  

$$v_t = \gamma v_{t-1}+\eta\Delta J(\theta_t;x,y)$$  

$$\theta_{t+1} = \theta_t - v_t$$  

$\gamma$: coefficient of momentum, 一般为0.9  

### 2. 牛顿加速梯度 Nesterov accelerated gradient(NAG)
效果比Momentum好点
思想：计算当前速度ｖ_t时,公式的后半部分，不再使用当前参数计算梯度，而是使用动量移动之后的参数计算梯度，相当于一个未来预测。
$$v_t = \gamma v_{t-1}+\eta\Delta J(\theta_t-\gamma v_{t-1})$$  

$$\theta_{t+1} = \theta_t - v_t$$  
![alt](imgs/nesterov.jpeg)
已知当前速度(图中绿色箭头)，计算移动速度之后的点(绿色箭头尖点)的梯度(红色箭头)，然后用来更新当前速度


## 自适应学习率优化算法 (Per-parameter adaptive learning rate methods)
自适应学习率优化算法针对于机器学习模型的学习率，传统的优化算法要么将学习率设置为常数要么根据训练次数调节学习率。极大忽视了学习率其他变化的可能性。然而，学习率对模型的性能有着显著的影响，因此需要采取一些策略来想办法更新学习率，从而提高训练速度。
### 1. Adagrad — Adaptive Gradient Algorithm
- 之前的方法都需要手动调参，使用同样的学习率，每次更新所有的参数
- 该方法自动适应每个参数的学习率,每个参数都有自己的学习率
- 对于具有较高梯度的网络参数，其学习率会变小；相反，较小的梯度或者低更新频率的参数，具有较大的学习率。因此Adagrad适用于数据稀疏或者分布不平衡的数据集（使某些参数的梯度为０，即该参数在本次不更新）
```
It adapts the learning rate to the parameters, performing smaller updates(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features
```
- 由于梯度的积累，最终学习率会变成０　　


对于网络中的第i个参数$\theta_i$,其梯度为
$$g_{i,t}=\nabla_\theta J(\theta_{t,i})$$
其更新公式为：
$$\theta_{i,t+1}=\theta_{t,i} - \frac{\eta}{\sqrt{G_{t,i}+\varepsilon}}g_{i,t}$$
其中$G_{t,i}$是该参数之前所有时刻梯度的平方和：
$$G_{t', i}=\sum_{t=1}^{t=t'} g_{i,t}^2$$

### 2. Adadelta
Adadelta是Adagrad的一种扩展算法，以处理Adagrad学习速率单调递减的问题。不是计算所有的梯度平方，Adadelta将计算计算历史梯度的窗口大小限制为一个固定值w。

在Adadelta中，无需存储先前的w个平方梯度，而是将梯度的平方递归地表示成所有历史梯度平方的均值。在t时刻的均值$E[g^2]_t$只取决于先前的均值和当前的梯度（分量γ类似于动量项）：
$$E[g^2]_t=\gamma E[g^2]_{t-1} + (1-\gamma)g_t^2$$　　

一般将γ设置成与动量项相似的值，即0.9左右
所以：
$$\Delta \theta_t = -\frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}} g_t$$
由于分母仅仅是梯度的均方根（root mean squared，RMS）误差，可以简写为：
$$\Delta \theta_t = -\frac{\eta}{RMS {[g]}_t} g_t$$

作者指出上述更新公式中的每个部分（与SGD，动量法或者Adagrad）并不一致，即更新规则中必须与参数具有相同的假设单位。为了实现这个要求，作者首次定义了另一个指数衰减均值，这次不是梯度平方，而是参数的平方的更新：
$$E[\Delta \theta^2]_t=\gamma E[\Delta \theta^2]_{t-1} + (1-\gamma)\Delta \theta^2_t$$　

因此，参数更新的均方根误差为：
$$RMS[\Delta \theta]_t = \sqrt{E[\Delta \theta^2]_t + \varepsilon}$$
由于$RMS[\Delta \theta]_t$是未知的(当前时刻的参数值的更新值)，所以用上一时刻的$RMS[\Delta \theta]_{t-1}$近似替代
最终更新公式为：
$$\Delta \theta_t=\frac{RMS[\Delta \theta]_{t-1}}{RMS[\Delta g]_{t}}g_t$$  

$$\theta_{t+1}=\theta_t+\Delta \theta_t$$  

### 3. RMSprop
RMSprop是先前我们得到的Adadelta的第一个更新向量的特例：
$$E[g^2]_t=\gamma E[g^2]_{t-1} + (1-\gamma)g_t^2$$　

$$\theta_{t+1}=\theta_t+\frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}}g_t$$  

建议将γ设置为0.9，对于学习率η，一个好的固定值为0.001

### 4. Adam — Adaptive Moment Estimation
Adam对每一个参数都计算自适应的学习率。除了像Adadelta和RMSprop一样存储一个指数衰减的历史平方梯度的平均$v_t$，Adam同时还保存一个历史梯度的指数衰减均值$m_t$，类似于动量:
$$m_t=β_1 m_{t−1}+(1−β_1)g_t$$

$$v_t=β_2 v_{t−1}+(1−β_2)g^2_t$$

mt 和vt分别是对梯度的一阶矩（均值）和二阶矩（非确定的方差）的估计，正如该算法的名称。当mt和vt初始化为0向量时，它们的值都偏向于0(biased to 0)，尤其是在初始化的步骤和当衰减率很小的时候（例如β1和β2趋向于1）。

通过计算偏差校正的一阶矩和二阶矩估计来抵消偏差：
$$\hat m_t=\frac{m_t}{1−β_{1,t}}$$

$$\hat v_t=\frac{v_t}{1−β_{2,t}}$$

更新公式为：
$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}\hat m_t$$ 

建议β1取默认值为0.9，β2为0.999

- 集合了adagrad和RMSprop的优点

### 5. Nadam- Nesterov-accelerated Adaptive Moment Estimation
将Nesterov momentum加入到Adam当中，即利用当前的Nesterov动量向量来代替Adam中的传统动量向量
一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。 

首先，原始的NAG公式为：
$$\theta_{t+1} = \theta_t -(\gamma m_{t-1}+\eta g_t)$$

将NAG进行如下改变，这里的直接使用下一步动量来更新参数：
$$g_t=\nabla_{\theta_t} J(\theta_t)$$

$$m_t=\gamma m_{t-1} + \eta g_t$$

$$\theta_{t+1} = \theta_t -(\gamma m_t+\eta g_t)$$



对于Adam，公式进行变形化简:
$$
\begin{aligned}
\theta_{t+1}& =\theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}\hat m_t\\ 
& = \theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}\hat\frac{m_{t}}{1-\beta_{t,1}}\\
& = \theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}(\frac{\beta_1 m_{t-1}}{1-\beta_{t,1}}+\frac{(1-\beta_1)g_t}{1-\beta_{t,1}})\\
& = \theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}(\beta_1 \hat m_{t-1}+\frac{(1-\beta_1)g_t}{1-\beta_{t,1}})\\
& = \theta_t-\frac{\eta}{\sqrt{\hat v_t} + \varepsilon}(\beta_1 \hat m_{t}+\frac{(1-\beta_1)g_t}{1-\beta_{t,1}})
\end{aligned}
$$
上面公式最后一步，就是根据变形后的NAG，将上一步的速度动量$m_{t-1}$替换成当前的$m_t$
