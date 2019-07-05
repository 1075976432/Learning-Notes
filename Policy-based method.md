# 基于策略的方法 Policy-Based Methods
区别：
* 基于值方法： 通过学习Q-value间接得到
* 基于策略方法： 直接学习最优策略
$$State \to^{networks} Actions$$
* 离散动作：输出对应每个动作的概率
* 连续动作： 输出动作数值  
优点：
* Simplicity
* Stochastic policies: 可以学习真正的随机策略(stochastic policies)
* Continuous action spaces: 适合连续空间动作

## Gradient Ascent
寻找网络参数$\theta$，最大化期望返回$J = J(\theta)$
简单的算法有：
* Hill Climbing
每次随机采样$\theta$,如果性能有提升，则更新$\theta$;否则，不更新
* Steepest Ascent Hill climbing
每次在当前$\theta$周围采样多个点，选取表现最好的更新
* Simulated annealing 
采样时，以某种方式，持续的减少exploration的范围
* Adaptive noise scaling 
decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.
采样时，当发现新的$\theta_{best}$时，减少搜索半径;否则增加
* Cross-Entropy Method
每次在当前$\theta$周围采样多个点，选取表现最好前$N%$的样本参数，取均值更新
* Evolution Strategies

