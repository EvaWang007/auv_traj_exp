# 一、核心结论：这篇论文为什么采用PSO

这篇论文中的PSO并不是用来替代Adam、反向传播，直接训练BiLSTM全部权重，而是作为一个**外层黑盒优化器**，自动搜索BiLSTM的网络结构和训练超参数；对每一个PSO粒子对应的候选配置，内部仍然使用Adam训练网络权重。

可以把论文的方法写成一个双层优化问题：

[
\lambda^*
=========

\arg\min_{\lambda\in\Lambda}
L_{\mathrm{val}}\left(\theta^*(\lambda),\lambda\right)
]

其中：

[
\theta^*(\lambda)
=================

\operatorname{AdamTrain}
\left(D_{\mathrm{train}},\lambda\right)
]

* (\lambda)：PSO搜索的网络结构和超参数；
* (\theta)：BiLSTM内部可训练权重和偏置；
* Adam负责优化(\theta)；
* PSO负责优化(\lambda)；
* 验证集MAPE作为PSO适应度；
* 测试集只用于最终性能评价。

论文明确指出，人工设计神经网络结构和调整超参数需要大量经验、时间和实验，且人工试验可能无法充分覆盖参数空间，因此作者借鉴AutoNN思想，用IPSO自动搜索BiLSTM结构和超参数。 

---

# 二、论文解决的具体问题

论文研究的是USV-AUV协同任务中的AUV短期轨迹预测。当水声通信发生中断时，USV需要根据AUV过去的状态预测其后续位置，以降低目标丢失风险。

模型的数据和训练设置如下。

| 项目           | 论文设置                          |
| ------------ | ----------------------------- |
| 输入           | 过去5个时刻的AUV状态                  |
| 每个时刻特征       | 纵向速度、横向速度、角速度、(x)、(y)、航向角，共6维 |
| 输出           | 下一个时刻的(x)或(y)坐标               |
| 数据量          | 5000个仿真数据点                    |
| 划分比例         | 训练集、验证集、测试集为7:1:2             |
| 训练方式         | (x)、(y)坐标分别训练                 |
| 固定窗口         | 5                             |
| 固定epoch      | 100                           |
| 固定batch size | 100                           |
| 网络优化器        | Adam                          |
| PSO种群规模      | 3                             |
| PSO迭代次数      | 10                            |
| PSO适应度       | 验证集MAPE                       |

论文使用滑动窗口将时间序列转换为形状为“样本数、时间步、特征数”的三维张量。 数据划分、输入输出形式和样本数量在实验部分也有明确说明。

---

# 三、论文实际搜索了哪些参数

论文给出的搜索范围是：

[
\begin{aligned}
N_{\text{layers}}&\in{1,2}\
N_{\text{neurons}}&\in[1,200]\
dropout&\in[0,0.5]\
learning\ rate&\in[0,0.2]
\end{aligned}
]

其中还分别提到了第一隐藏层神经元数和第二隐藏层神经元数。 

作者声称每个粒子是“六维”的，但正文实际明确列出的项目只有：

1. BiLSTM隐藏层数；
2. 第一隐藏层节点数；
3. 第二隐藏层节点数；
4. Dropout；
5. 学习率。

也就是文字上只能明确对应到五个维度。这是论文描述中的一个明显不一致，可能遗漏了一个参数，也可能是“six dimensions”的笔误。

## 搜索空间还存在两个未说明的问题

第一，层数和神经元数是离散整数，而Dropout和学习率是连续变量。论文没有说明：

* 整数参数如何取整；
* 当隐藏层数为1时，第二层节点数如何处理；
* 粒子越界时如何修复；
* 学习率是否使用线性尺度还是对数尺度；
* 是否缓存重复配置。

第二，学习率搜索范围设为0到0.2并不理想：

* 学习率为0时网络完全无法更新；
* 对Adam而言，0.2通常是非常激进的上界；
* 学习率往往跨越多个数量级，更适合搜索 (\log_{10}(lr))，例如 (10^{-5}) 到 (10^{-2})，而不是在线性空间内搜索。

---

# 四、论文使用PSO的六个主要原因

## 1. 神经网络超参数优化是黑盒问题

给定一个BiLSTM配置后，必须实际构建并训练模型，才能知道验证误差。一般无法写出如下显式函数：

[
MAPE=f(N_{\text{layers}},N_{\text{neurons}},dropout,lr)
]

更无法直接计算：

[
\frac{\partial MAPE}{\partial N_{\text{layers}}}
]

因为隐藏层数量和神经元数量本身就是离散变量。

PSO不需要目标函数的解析形式和梯度，只需要输入一个配置，训练模型，再返回验证误差。因此它适合这种昂贵的黑盒优化。

## 2. 搜索空间是离散变量和连续变量的混合空间

论文同时搜索：

* 离散变量：隐藏层数、节点数；
* 连续变量：Dropout、学习率。

传统梯度法无法直接处理“隐藏层数从1变成2”这样的结构变化。PSO可以将这些变量统一编码成粒子位置，再通过取整或映射得到实际配置。

## 3. 网络性能关于超参数通常是非凸、多峰的

神经网络验证误差会受到：

* 模型容量；
* 学习率；
* 正则化强度；
* 随机初始化；
* mini-batch顺序；
* 训练轮数；

等因素共同影响。

因此超参数到验证误差的映射通常不平滑，也可能存在多个局部优良区域。PSO通过多个粒子同时探索不同区域，并同时利用个体最优 `pbest` 和群体最优 `gbest`，比单点局部搜索更适合这类多峰问题。PSO最初就是作为随机的群体非线性优化方法提出的，其原始论文也将神经网络训练列为潜在应用。([数字对象识别][1])

## 4. 相比人工试验，PSO能够自动利用历史搜索结果

随机搜索的每一次试验相互独立，而PSO粒子会利用：

* 自己过去找到的最好位置；
* 整个种群找到的最好位置。

因此后续配置不是完全盲目采样，而是在已有优良配置周围进一步搜索。

不过，这并不意味着PSO一定比随机搜索更好。随机搜索是超参数优化必须比较的重要基线，在很多只有少数参数真正重要的任务中，随机搜索明显优于规则网格搜索。([机器学习研究杂志][2])

## 5. PSO结构简单，便于并行评估

不同粒子的候选神经网络通常可以在不同GPU或不同进程上独立训练。PSO天然适合批量并行候选评估。已有PSO与贝叶斯优化的比较研究也专门分析了PSO对并行计算资源的利用能力。([科学直通车][3])

## 6. PSO能够同时优化结构与训练参数

论文的目标不是仅调整学习率，而是希望同时确定：

* BiLSTM有几层；
* 每层有多少神经元；
* Dropout多大；
* 学习率多大。

这属于简化的神经架构搜索，即AutoNN或NAS问题。PSO的粒子编码可以将这些变量放在同一个候选向量中进行联合搜索。

---

# 五、为什么论文不直接使用传统PSO，而使用IPSO

标准PSO速度更新公式是：

[
v_i^{t+1}
=========

wv_i^t+
c_1r_1(pbest_i-x_i^t)+
c_2r_2(gbest-x_i^t)
]

位置更新为：

[
x_i^{t+1}=x_i^t+v_i^{t+1}
]

其中：

* (w)：惯性权重；
* (c_1)：个体学习因子；
* (c_2)：社会学习因子；
* (r_1,r_2)：0到1之间的随机数。

论文认为固定惯性权重存在矛盾：

* (w)较大时，粒子移动步长较大，有利于全局探索，但局部搜索精度下降；
* (w)较小时，粒子移动步长较小，有利于局部精细搜索，但容易过早聚集并错过其他优良区域。

这就是PSO经典的：

[
\text{Exploration}
\quad\text{与}\quad
\text{Exploitation}
]

之间的平衡问题。

## 论文提出的动态惯性权重

作者从二阶系统阻尼特性出发，定义：

[
\phi_1=c_1r_1,\qquad
\phi_2=c_2r_2
]

令：

[
s=\phi_1+\phi_2
]

最终惯性权重为：

[
w=1+s-\sqrt{s(s+2)}
]

作者采用的阻尼比为：

[
\xi=\frac{\sqrt{2}}{2}
]

并声称这种动态惯性权重可以在前期快速接近全局极值，在后期快速收敛。

如果采用论文所说的：

[
c_1=c_2=2
]

因为 (r_1,r_2\in(0,1))，所以：

[
s\in(0,4)
]

相应地：

[
w\in(5-\sqrt{24},1)
\approx(0.101,1)
]

其机制可以理解为：

* 当认知项和社会项的吸引强度较弱，即 (s)较小时，(w)较大，粒子更依赖原有速度；
* 当吸引强度较强，即 (s)较大时，(w)较小，粒子更倾向于靠近 `pbest` 和 `gbest`。

## 对这一改进的技术性判断

这个公式确实实现了“吸引项越强，惯性项越弱”的协调控制。

但是，公式本身没有显式包含迭代次数 (t)。因此它并不能从数学形式上保证：

[
w_{\text{early}}>w_{\text{late}}
]

也就是不能天然保证严格的“前期大惯性、后期小惯性”。除非实际代码还对 (c_1,c_2)、随机因子或其他变量进行了随迭代变化的调度，而论文没有提供这部分实现细节。

所以更准确的表述应该是：

> 论文提出的是随当前认知和社会吸引强度变化的随机自适应惯性权重，而不是传统意义上由迭代次数直接控制的单调递减惯性权重。

惯性权重的概念最早由Shi和Eberhart引入，用于调节PSO的全局探索和局部开发能力。([ResearchGate][4]) Clerc和Kennedy则从动力系统稳定性角度分析了粒子轨迹，并提出了控制收敛趋势的系数模型和收缩因子思想。([IEEE Xplore][5])

---

# 六、论文中的完整IPSO-BiLSTM过程

论文的算法流程可以还原为：

```text
初始化一组粒子
每个粒子表示一个BiLSTM配置
        ↓
对于每个粒子：
    解码隐藏层数、节点数、Dropout和学习率
    使用训练集训练BiLSTM
    使用验证集预测
    计算验证集MAPE
    将MAPE作为粒子适应度
        ↓
更新粒子的个体最优pbest
更新整个种群的全局最优gbest
        ↓
根据IPSO公式更新速度、位置和惯性权重
        ↓
重复10轮
        ↓
将gbest对应的配置作为最终BiLSTM配置
        ↓
用测试集评价最终模型
```

论文明确说明：当前粒子确定BiLSTM模型后，先在训练集训练，再在验证集预测，并将验证集MAPE作为粒子适应度。

其伪代码也明确包含初始化位置与速度、计算验证适应度、更新 `pbest`、`gbest`、速度、位置和惯性权重。

---

# 七、实验结果是否证明了使用PSO的必要性

论文第6页表III给出了四种模型的结果。

以RMSE为例：

| 模型          | (x) RMSE | (y) RMSE |
| ----------- | -------: | -------: |
| LSTM        |  42.4847 |  39.1048 |
| BiLSTM      |  36.5255 |  29.8080 |
| PSO-BiLSTM  |  28.7803 |  17.8359 |
| IPSO-BiLSTM |  27.1634 |  12.6480 |

根据表中数值计算：

### PSO-BiLSTM相对于普通BiLSTM

[
x\text{方向RMSE下降约 }21.2%
]

[
y\text{方向RMSE下降约 }40.2%
]

这表明，在论文的实验条件下，自动搜索结构和超参数比固定的“1层、64节点、学习率0.001、Dropout 0.2”配置表现更好。

### IPSO-BiLSTM相对于标准PSO-BiLSTM

[
x\text{方向RMSE进一步下降约 }5.6%
]

[
y\text{方向RMSE进一步下降约 }29.1%
]

因此实验结果支持两个结论：

1. PSO超参数搜索对结果有帮助；
2. 自适应惯性权重可能进一步改善搜索结果。

但是，这些结果还不足以严格证明IPSO本身具有普遍优势。

---

# 八、这篇论文在实验设计上的主要不足

## 1. PSO搜索预算太小

论文设置：

[
population=3,\qquad iterations=10
]

理论上最多只评估：

[
3\times10=30
]

个候选配置。

而节点数、学习率、Dropout等组成的搜索空间非常大。30次评估很难证明充分搜索了该空间。

## 2. 没有重复随机实验

神经网络训练和PSO都具有随机性。论文没有说明：

* 是否使用多个随机种子；
* 每个候选是否训练多次；
* 表III是单次结果还是多次平均；
* 是否报告标准差和置信区间；
* 是否进行了显著性检验。

如果每种方法只运行一次，那么性能差异可能部分来自随机初始化和训练顺序。

## 3. 没有与随机搜索和贝叶斯优化做等预算比较

只比较：

* 手工固定BiLSTM；
* PSO-BiLSTM；
* IPSO-BiLSTM；

并不能说明PSO是最合适的超参数优化器。

合理的实验应当在相同的：

* 候选模型评估次数；
* GPU时长；
* 总训练epoch；
* 随机种子；

条件下比较随机搜索、PSO、贝叶斯优化和多保真方法。

随机搜索是必要基线；贝叶斯优化通常更强调昂贵黑盒目标下的样本效率；Hyperband则通过早停和自适应资源分配，减少在差配置上浪费的训练资源。([机器学习研究杂志][2])

## 4. 使用MAPE作为轨迹优化目标存在局限

论文使用：

[
MAPE=
\frac1N\sum_t
\left|
\frac{\hat p(t)-p(t)}{p(t)}
\right|
]

作为PSO适应度。

问题在于：

* 当真实坐标接近0时，MAPE会不稳定；
* MAPE依赖坐标原点；
* 如果整体坐标带有很大的常数偏移，即使绝对轨迹误差较大，MAPE也可能较小；
* 分别计算(x)、(y)误差，不能直接表示二维空间距离；
* 它不能专门衡量预测终点误差和长预测区间误差。

论文第5页图中的绝对坐标大约位于数百到一千米量级，因此使用绝对坐标作为MAPE分母，可能使百分比误差看起来较小。

对轨迹预测，更建议使用：

[
ADE=\frac1{NT}
\sum_{i=1}^{N}\sum_{t=1}^{T}
|\hat{\mathbf p}*{i,t}-\mathbf p*{i,t}|_2
]

以及：

[
FDE=\frac1N
\sum_{i=1}^{N}
|\hat{\mathbf p}*{i,T}-\mathbf p*{i,T}|_2
]

这也是你之前代码中的适应度：

[
ADE+0.5FDE
]

比论文的单坐标MAPE更符合二维、多步轨迹预测目标。

## 5. (x)、(y)分别训练忽略了坐标间相关性

论文将(x)和(y)坐标分别训练。

但二维运动满足耦合关系：

[
\Delta x_t=v_t\cos\psi_t\Delta t
]

[
\Delta y_t=v_t\sin\psi_t\Delta t
]

独立预测可能无法充分利用：

* 航向角与坐标变化的耦合；
* 速度与位移之间的物理关系；
* (x)、(y)轨迹的联合分布。

联合输出二维位置或二维速度通常更自然。

## 6. 数据完全来自仿真

论文数据由LOS制导和PID控制的AUV仿真模型生成，而非真实海试数据。 作者也在结论中将获取真实AUV航行数据列为后续工作。

因此目前结果主要证明模型能够拟合该仿真系统，尚不能直接证明其对：

* 海流扰动；
* 传感器噪声；
* 定位漂移；
* 控制器变化；
* 不同AUV动力学；

具有良好泛化能力。

## 7. 没有报告搜索成本

论文没有给出：

* 一次候选训练耗时；
* 30次搜索总耗时；
* PSO与IPSO的计算开销差异；
* 获得单位精度提升所需的GPU时间。

对于神经网络超参数优化，最终精度并不是唯一指标，还应报告：

[
\text{验证性能}\quad+\quad
\text{搜索时间}\quad+\quad
\text{模型复杂度}
]

## 8. 论文中存在文字不一致

除了“六维粒子只列出五个变量”外，结论中写的是优于“PSO-LSTM”，但实验表和正文比较的是“PSO-BiLSTM”，应当是笔误。

---

# 九、这篇论文与之前代码的关系

你之前的代码受到类似思想影响，但它并不是论文中严格意义上的IPSO实现。

| 对比项   | 论文                    | 之前代码                                     |
| ----- | --------------------- | ---------------------------------------- |
| 网络    | BiLSTM                | 单向LSTM                                   |
| 搜索方法  | 自适应惯性权重IPSO           | 固定惯性权重标准PSO                              |
| 惯性权重  | 根据(\phi_1,\phi_2)动态计算 | 固定 `w=0.55`                              |
| 搜索参数  | 层数、两层节点数、Dropout、学习率  | hidden_dim、Dropout、学习率、batch size、4个损失权重 |
| 适应度   | 验证集MAPE               | `ADE + 0.5 × FDE`                        |
| 种群与迭代 | 3粒子、10轮               | 默认5粒子、5轮                                 |
| 候选训练  | 100 epoch，论文描述较模糊     | 搜索阶段6 epoch                              |
| 重复配置  | 未说明                   | 有离散配置缓存                                  |
| 最终任务  | 单步分别预测(x)、(y)         | 多步联合二维轨迹预测                               |

你的代码使用固定PSO参数：

```python
w = 0.55
c1 = 1.4
c2 = 1.4
```

并按标准公式更新速度和位置，因此不能称为论文公式对应的IPSO。 

不过，你的代码在评价指标方面更适合轨迹任务，因为它直接优化：

[
ADE+0.5FDE
]



你的代码还搜索了物理一致性和多任务损失权重，这是论文没有涉及的：

* `alpha_cur`；
* `alpha_bias`；
* `alpha_pi`；
* `beta_smooth`。

---

# 十、PSO优化神经网络可以分成四类研究

学习这一领域时，首先要区分PSO到底在优化什么。

## 第一类：PSO直接优化神经网络权重

粒子位置直接表示网络全部权重和偏置：

[
x_i=
[w_1,w_2,\ldots,w_D,b_1,\ldots,b_K]
]

适应度是训练损失或验证损失。

优点是不依赖梯度，可以跳出部分局部区域；缺点是现代深度网络参数可能达到百万甚至亿级，粒子维度过高，评估成本很大。

## 第二类：PSO只优化初始权重，随后使用反向传播

流程为：

[
\text{PSO寻找较优初始化}
\rightarrow
\text{Adam/BP局部精调}
]

这种方式发挥：

* PSO的全局搜索能力；
* 梯度法的局部高效收敛能力。

## 第三类：PSO优化超参数

这是上传论文和你之前代码采用的方式。

粒子表示：

[
x_i=
[\text{learning rate},
\text{hidden size},
\text{dropout},
\text{batch size},\ldots]
]

每个粒子的适应度需要通过训练一个候选网络获得。

## 第四类：PSO进行神经架构搜索

粒子编码完整网络结构，例如：

* 层类型；
* 层数；
* 每层宽度；
* 卷积核大小；
* 跳跃连接；
* 激活函数；
* 池化方式。

这种方法属于进化神经架构搜索，搜索空间和计算成本通常远高于普通超参数搜索。

---

# 十一、建议按以下文献路线系统学习

## 阶段1：PSO基础理论

### 1. Kennedy & Eberhart, 1995

**Particle Swarm Optimization**

这是PSO原始论文。重点理解：

* 粒子位置与速度；
* `pbest`和`gbest`；
* 社会认知模型；
* PSO与遗传算法的区别。

DOI：`10.1109/ICNN.1995.488968`。([数字对象识别][1])

### 2. Shi & Eberhart, 1998

**A Modified Particle Swarm Optimizer**

重点理解惯性权重(w)为什么被引入，以及：

* 大(w)：更强调探索；
* 小(w)：更强调局部开发；
* 为什么常使用递减惯性权重。

DOI：`10.1109/ICEC.1998.699146`。([ResearchGate][4])

### 3. Clerc & Kennedy, 2002

**The Particle Swarm—Explosion, Stability, and Convergence in a Multidimensional Complex Space**

这是理解PSO稳定性、粒子爆炸和收缩因子的核心论文。重点看：

* 粒子轨迹的动力系统分析；
* 什么参数组合可能导致速度发散；
* constriction factor；
* 惯性权重和收缩因子的关系。

DOI：`10.1109/4235.985692`。([OA Monitor Ireland][6])

### 4. Van den Bergh & Engelbrecht, 2004

**A Cooperative Approach to Particle Swarm Optimization**

重点理解高维优化为什么困难，以及如何把一个高维向量拆分给多个子种群协同优化。这对直接用PSO优化神经网络权重尤其重要。

DOI：`10.1109/TEVC.2004.826069`。([ResearchGate][7])

---

## 阶段2：PSO直接训练神经网络

### 5. Meissner et al., 2006

**Optimized Particle Swarm Optimization and Its Application to Artificial Neural Network Training**

这篇论文同时优化PSO自身参数，并用PSO训练神经网络。重点理解：

* PSO自身也有需要调节的元参数；
* “用PSO优化PSO”的元优化思想；
* 神经网络权重训练与超参数优化的区别；
* 验证集如何参与模型选择。

研究发现，针对任务优化PSO参数后，可以降低神经网络训练所需的迭代数。([Springer][8])

### 6. PSO-Trained Recurrent Neural Network, 2018

**Particle Swarm Optimization Trained Recurrent Neural Network for Voltage Instability Prediction**

适合观察PSO直接训练RNN权重与传统反向传播的差异。重点关注：

* 粒子如何编码循环网络参数；
* 高维权重搜索的成本；
* PSO和BP的性能及收敛比较。

DOI：`10.1016/j.jesit.2017.05.001`。([科学直通车][9])

---

## 阶段3：PSO优化神经网络结构和超参数

### 7. Ye, 2017

**Particle Swarm Optimization-Based Automatic Parameter Selection for Deep Neural Networks**

这篇论文非常适合作为“外层PSO＋内层梯度训练”的入门文献。它将网络配置编码为实数粒子，PSO搜索网络结构，梯度下降对每个候选网络进行少量训练。([PLOS][10])

重点学习：

* 双层优化结构；
* 低保真候选评估；
* 少量epoch近似最终性能；
* 连续粒子如何编码离散结构。

### 8. cPSO-CNN, 2019

**An Efficient PSO-Based Algorithm for Fine-Tuning Hyper-Parameters of Convolutional Neural Networks**

论文针对固定架构CNN进行超参数微调，并针对不同超参数取值范围重新设计了加速系数。重点理解：

* 不同维度量纲差异；
* 专家先验如何注入PSO；
* PSO在深度模型超参数调优中的探索效率。

DOI：`10.1016/j.swevo.2019.06.002`。([科学直通车][11])

### 9. MPSO-CNN, 2021

**Hybrid MPSO-CNN: Multi-Level Particle Swarm Optimized Hyperparameters of CNN**

该方法使用两级粒子群：

* 第一层搜索网络架构；
* 第二层搜索训练超参数。

适合理解“架构搜索”和“超参数搜索”为什么可以分层进行，以及为什么单一粒子向量可能难以同时处理两类变量。

DOI：`10.1016/j.swevo.2021.100863`。([科学直通车][12])

### 10. Xu et al., 2022

**Research on Particle Swarm Optimization in LSTM Neural Networks for Rainfall-Runoff Simulation**

这篇与AUV论文最接近，都是：

* 时间序列；
* LSTM；
* PSO超参数优化；
* 预测任务。

论文还分析了不同预测提前期下的最佳超参数组合，适合学习“最佳超参数可能随预测时域变化”的问题。

DOI：`10.1016/j.jhydrol.2022.127553`。([科学直通车][13])

### 11. IPSO-LSTM Stock Forecast, 2021

**A Novel Improved Particle Swarm Optimization With Long-Short Term Memory Hybrid Model for Stock Indices Forecast**

该文使用：

* 非线性惯性权重；
* 自适应变异；
* IPSO优化LSTM超参数。

适合与上传论文的自适应惯性机制做对比，尤其关注如何缓解过早收敛。

DOI：`10.1109/ACCESS.2021.3056713`。([数字对象识别][14])

---

## 阶段4：PSO神经架构搜索

### 12. Kiranyaz et al., 2009

**Evolutionary Artificial Neural Networks by Multi-Dimensional Particle Swarm Optimization**

其关键贡献是让粒子不仅搜索参数位置，还能搜索问题维度，也就是自动决定网络规模。适合理解：

* 可变维度粒子；
* 网络拓扑编码；
* 如何自动确定隐藏节点和网络结构。

DOI：`10.1016/j.neunet.2009.05.013`。([科学直通车][15])

### 13. Cooperative Binary-Real PSO, 2011

**Tuning the Structure and Parameters of a Neural Network Using Cooperative Binary-Real PSO**

该方法同时使用：

* 二进制变量表示连接是否存在；
* 实数变量表示网络权重或参数。

它非常适合学习混合搜索空间编码。

DOI：`10.1016/j.eswa.2010.09.154`。([科学直通车][16])

### 14. Multiobjective PSO-NAS, 2020

**Efficient Network Architecture Search via Multiobjective PSO Based on Decomposition**

其重点不只是最小化预测误差，还同时考虑模型复杂度。适合学习：

[
\min
\left(
\text{validation error},
\text{parameter count}
\right)
]

这比只追求精度更适合部署在AUV、USV等计算和能源受限平台。

DOI：`10.1016/j.neunet.2019.12.005`。([科学直通车][17])

---

## 阶段5：混合PSO和梯度优化

### 15. PSO-PINN, 2022

**PSO-PINN: Physics-Informed Neural Networks Trained with Particle Swarm Optimization**

这篇预印本提出PSO-BP-CD：

* 前期更多依赖PSO进行全局搜索；
* 后期逐渐增加反向传播的权重；
* 将PSO和局部梯度优化结合。

这与你目前的物理一致性LSTM研究很相关，因为你的损失中已经包含物理积分项和多任务约束。([arXiv][18])

---

# 十二、还必须学习的三类非PSO基线

要严谨研究PSO超参数优化，不能只看PSO论文。

## 1. 随机搜索

Bergstra和Bengio证明，在很多超参数问题中，真正重要的参数只有少数几个，因此随机搜索通常比网格搜索更有效。它是PSO实验不可缺少的基线。([机器学习研究杂志][2])

## 2. 贝叶斯优化

贝叶斯优化通过代理模型近似“超参数到验证性能”的关系，更强调昂贵候选评估条件下的样本效率。([NeurIPS 会议论文集][19])

一般来说：

* 候选评估非常昂贵；
* 维度较低；
* 总评估次数较少；

时，贝叶斯优化往往很有竞争力。

## 3. Hyperband或Successive Halving

这类方法不是重点决定“下一组参数往哪里搜索”，而是决定：

> 哪些候选值得继续投入更多训练资源。

它可以在早期淘汰明显较差的配置，避免所有粒子都完整训练100个epoch。([机器学习研究杂志][20])

---

# 十三、推荐的系统学习与复现实验路线

## 第一阶段：实现标准PSO

先不要接入神经网络，使用以下测试函数：

* Sphere；
* Rosenbrock；
* Rastrigin；
* Ackley。

研究：

* (w)；
* (c_1,c_2)；
* 粒子数量；
* 速度上限；
* 全局拓扑与局部拓扑；
* 不同边界处理方式。

输出：

* 最优适应度曲线；
* 种群多样性曲线；
* 粒子轨迹；
* 多随机种子均值和标准差。

## 第二阶段：比较不同惯性策略

至少实现：

1. 固定惯性权重；
2. 线性递减惯性权重；
3. 随机惯性权重；
4. 论文中的阻尼协调惯性权重；
5. Clerc收缩因子；
6. 你的固定 `w=0.55`。

在相同函数、相同评估次数和相同随机种子上比较。

## 第三阶段：用PSO直接训练一个小型MLP

选择参数量很小的网络，例如：

[
2\rightarrow8\rightarrow1
]

比较：

* Adam；
* 标准PSO；
* IPSO；
* PSO初始化＋Adam精调。

这一阶段用于理解“优化权重”和“优化超参数”的本质区别。

## 第四阶段：用PSO优化LSTM超参数

先限定为4个参数：

[
[\log_{10}(lr),hidden_dim,dropout,batch_size]
]

建议范围：

[
\log_{10}(lr)\in[-5,-2]
]

[
hidden_dim\in{32,64,96,128,192,256}
]

[
dropout\in[0,0.5]
]

[
batch_size\in{32,64,128,256}
]

同时比较：

* 随机搜索；
* 标准PSO；
* IPSO；
* 贝叶斯优化。

严格保证每种方法拥有相同的候选模型评估次数。

## 第五阶段：加入多保真候选评估

可以采用：

* 第一级：20%训练数据，3个epoch；
* 第二级：50%训练数据，10个epoch；
* 第三级：全部训练数据，完整训练。

只有表现较好的候选进入下一阶段。

这可以显著降低PSO搜索成本，也能减少“所有粒子都训练完整模型”的资源浪费。

## 第六阶段：复现并改进AUV论文

建议按照以下顺序进行消融实验：

| 实验             | 目的          |
| -------------- | ----------- |
| 固定BiLSTM       | 基线          |
| 随机搜索BiLSTM     | 判断自动搜索本身的贡献 |
| 标准PSO-BiLSTM   | 判断PSO贡献     |
| 线性递减PSO-BiLSTM | 比较经典自适应惯性   |
| 论文IPSO-BiLSTM  | 判断阻尼公式贡献    |
| IPSO＋ADE/FDE   | 判断适应度函数贡献   |
| IPSO＋物理约束      | 判断物理先验贡献    |
| IPSO＋多保真训练     | 判断搜索效率改进    |

---

# 十四、严谨实验中必须控制的变量

PSO神经网络论文最容易出现的问题是：把随机波动误认为优化算法提升。

推荐每个候选配置的适应度定义为：

[
fitness(\lambda)
================

\frac1K
\sum_{k=1}^{K}
L_{\mathrm{val}}^{(k)}(\lambda)
]

其中 (K) 是不同随机种子数量，建议至少：

[
K=3
]

最终比较至少运行5次完整PSO搜索，并报告：

[
\text{mean}\pm\text{std}
]

还应统一：

* 数据划分；
* 数据预处理；
* 候选训练epoch；
* early stopping规则；
* 总模型评估次数；
* 总GPU时长；
* 初始化种子集合；
* 测试集使用次数。

测试集不应参与：

* PSO适应度；
* early stopping；
* 搜索范围修改；
* 模型选择。

---

# 十五、对你当前研究最有价值的改进方向

结合这篇论文和你之前的代码，最值得研究的不是简单扩大粒子数，而是以下组合：

[
\boxed{
\text{混合离散IPSO}
+
\text{多保真候选评估}
+
\text{多随机种子适应度}
+
\text{ADE/FDE轨迹目标}
+
\text{物理一致性约束}
}
]

具体来说：

1. 把固定 `w=0.55` 改为可切换的惯性策略；
2. 学习率改为对数空间搜索；
3. 对整数和类别变量设计明确的解码规则；
4. 将重复组合缓存扩展为“配置＋种子＋训练预算”缓存；
5. 用多个随机种子的平均验证ADE/FDE作为适应度；
6. 加入训练时间或参数量作为第二目标；
7. 与随机搜索和贝叶斯优化做等预算比较；
8. 最终在测试集上只评价一次全局选定方案。

这条路线可以把目前的“PSO调参代码”提升为一套实验设计较完整、具备论文研究价值的神经网络自动优化框架。

[1]: https://doi.org/doi%3A%2010.1109/ICNN.1995.488968?utm_source=chatgpt.com "Particle swarm optimization"
[2]: https://www.jmlr.org/papers/v13/bergstra12a.html?utm_source=chatgpt.com "Random Search for Hyper-Parameter Optimization"
[3]: https://www.sciencedirect.com/science/article/pii/S0010465523003004?utm_source=chatgpt.com "Comparison of Bayesian and particle swarm algorithms for hyperparameter optimisation in machine learning applications in high energy physics - ScienceDirect"
[4]: https://www.researchgate.net/publication/3755900_A_Modified_Particle_Swarm_Optimizer?utm_source=chatgpt.com "(PDF) A Modified Particle Swarm Optimizer"
[5]: https://ieeexplore.ieee.org/abstract/document/985692?utm_source=chatgpt.com "The particle swarm - explosion, stability, and convergence in a multidimensional complex space"
[6]: https://oamonitor.ireland.openaire.eu/rpo/rcsi/search/publication?pid=10.1109%2F4235.985692&utm_source=chatgpt.com "The particle swarm - explosion, stability, and convergence in a multidimensional complex space"
[7]: https://www.researchgate.net/publication/3418789_A_Cooperative_Approach_to_Particle_Swarm_Optimization?utm_source=chatgpt.com "(PDF) A Cooperative Approach to Particle Swarm Optimization"
[8]: https://link.springer.com/article/10.1186/1471-2105-7-125?utm_source=chatgpt.com "Optimized Particle Swarm Optimization (OPSO) and its application to artificial neural network training | BMC Bioinformatics | Springer Nature Link"
[9]: https://www.sciencedirect.com/science/article/pii/S231471721730020X?utm_source=chatgpt.com "Particle Swarm Optimization trained recurrent neural network for voltage instability prediction - ScienceDirect"
[10]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0188746&utm_source=chatgpt.com "Particle swarm optimization-based automatic parameter selection for deep neural networks and its applications in large-scale and high-dimensional data | PLOS One"
[11]: https://www.sciencedirect.com/science/article/pii/S2210650218310083?utm_source=chatgpt.com "cPSO-CNN: An efficient PSO-based algorithm for fine-tuning hyper-parameters of convolutional neural networks - ScienceDirect"
[12]: https://www.sciencedirect.com/science/article/pii/S2210650221000249?utm_source=chatgpt.com "Hybrid MPSO-CNN: Multi-level Particle Swarm optimized hyperparameters of Convolutional Neural Network - ScienceDirect"
[13]: https://www.sciencedirect.com/science/article/pii/S0022169422001287?utm_source=chatgpt.com "Research on particle swarm optimization in LSTM neural networks for rainfall-runoff simulation - ScienceDirect"
[14]: https://doi.org/10.1109/ACCESS.2021.3056713?utm_source=chatgpt.com "A Novel Improved Particle Swarm Optimization With Long-Short Term Memory Hybrid Model for Stock Indices Forecast"
[15]: https://www.sciencedirect.com/science/article/pii/S0893608009001038?utm_source=chatgpt.com "Evolutionary artificial neural networks by multi-dimensional particle swarm optimization - ScienceDirect"
[16]: https://www.sciencedirect.com/science/article/pii/S0957417410011115?utm_source=chatgpt.com "Tuning the structure and parameters of a neural network using cooperative binary-real particle swarm optimization - ScienceDirect"
[17]: https://www.sciencedirect.com/science/article/pii/S0893608019303971?utm_source=chatgpt.com "Efficient network architecture search via multiobjective particle swarm optimization based on decomposition - ScienceDirect"
[18]: https://arxiv.org/abs/2202.01943?utm_source=chatgpt.com "PSO-PINN: Physics-Informed Neural Networks Trained with Particle Swarm Optimization"
[19]: https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html?utm_source=chatgpt.com "Practical Bayesian Optimization of Machine Learning Algorithms"
[20]: https://www.jmlr.org/beta/papers/v18/16-558.html?utm_source=chatgpt.com "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
