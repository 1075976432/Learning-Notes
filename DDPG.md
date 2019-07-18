## Deep Deterministic Policy Gradient (DDPG) 
#### Deterministic Policy Gradient
* 结合了actor-critic和DQN结构,拥有replay buffer
* 使用deterministic policy


##### Deterministic Policy Gradient
直观上来说，我们应该朝着使得值函数Q值增大的方向去更新策略的参数θ,因此算法根据state value来更新policy，所以：
$$g=\nabla_\theta Q_w(s, \pi_\theta (s))$$

根据链式法则：
$$g=\nabla_a Q_w(s, a) \nabla_\theta \pi_\theta (s)$$  
##### loss function
假设：
critic网络：$Q(s,a|\theta^Q)$
critic target网络：$Q(s,a|\theta^{Q'})$
actor网络：$\mu(s|\theta^\mu)$
actor target网络：$\mu(s|\theta^{\mu'})$

对于critic：
$$Loss = \frac {1}{N} \sum^N_i [r_i+\gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})| \theta^{Q'})-Q(s_i,a_i| \theta^Q)]^2$$
对于actor：
$$\nabla_{\theta^\mu J} \approx \frac {1}{N} \sum^N_i \nabla_a Q(s, a|\theta^Q) \nabla_{\theta^\mu} \mu (s|\theta^\mu )$$

