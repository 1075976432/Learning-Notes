<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>  

# Deep Q-Networks  
基本思想：随着state的增多，action的增多，或者state或者action为连续空间值，建立Q-table将不再合适。一种解决方法就是：用神经网络代替Q-table，其接受state输出所有动作的Q值。
## paper分析
用神经网络预测Q值  
1. state：为了减少数量，进行一下操作：  
彩图转灰度图； 
图像大小减少；  
由于帧与帧之间存在顺序关系，所以一个state由4帧组成。  
2. Q value  
通过CNN网络生成每个动作的Q value  
![avatar](./imgs/cnn_config.png)
3. Experience Replay
通过建立database of samples,并从其中学习mapping，将强化学习问题转化成了监督学习。
* 原因：
    * naive Q-learning使用一组 <S, A, S', R> 后直接抛弃，效率不高，因此可以将所有**经验**存储在replay buffer，并在之后的学习中使用
    * 由于经验之间存在强相关(highly correlated),naive Q-learning面临着Q不收敛的风险
* 解决：
    * 在训练初始阶段，只采集经验，并不学习，尽可能探索。
    * 从replay beffer中随机采样，避免了经验间的相关性。
4. Fixed-Target  
首先考虑Q-learning update,其损失函数为
    $$E_\pi [q_\pi(S,A) - \hat q(S,A,w))^2]$$
求导得到梯度：
        $$\nabla_w J(w)=-2(q_\pi(S,A) - \hat q(S,A,w))\nabla_w \hat q(S,A,w)$$
w的更新值为：
$$
\begin{aligned}\Delta w
&=-\alpha \frac{1}{2} \nabla_w J(w) \\
&=\alpha (q_\pi(S,A) - \hat q(S,A,w))\nabla_w \hat q(S,A,w)
\end{aligned}
$$
由上式可看出，$q_\pi$是需要学习的optimal policy,称作TD Target，其应该是与网络参数$w$无关。但实际在D-learning中，其w的更新值为：
    $$\Delta w=\alpha (R+\gamma max\hat q(S',a, w) - \hat q(S,A,w))\nabla_w \hat q(S,A,w)$$
TD Target由$R+\gamma max\hat q(S',a, w)$代替。由于其本身与需要学习的参数$w$有关，并不是真实的optimal policy。所以相当于用一个猜测的值来更新另外一个猜测的值，就像'carrot stick ride',其可能导致不收敛。
![avatar](./imgs/q-up.png)
* 解决方法
在网络学习(learning step)时，固定TD Target。具体做法，构建另一个一样的网络，称作目标网络(target network)，其参数为$w^-$，并且在网络学习时固定不变，作为TD Target。在经过一段时间后，更新$w^-$。
![avatar](./imgs/q-up2.png)

### reference
- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529. http://www.davidqiu.com:8888/research/nature14236.pdf