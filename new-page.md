---
title: 测试
description: 
published: 1
date: 2022-11-12T18:38:16.937Z
tags: 
editor: markdown
dateCreated: 2022-11-12T18:34:35.598Z
---

### 参考文档

https://blog.csdn.net/Pysamlam/article/details/123038845

https://zhuanlan.zhihu.com/p/306826637


https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/index.html


https://zhuanlan.zhihu.com/p/53526998

quant 工具箱
https://zhuanlan.zhihu.com/p/77022897



![486444c4a1cdce11956364e46bbb7cfd.jpeg](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p354)

rust kafaka 计算框架

https://github.com/vertexclique/callysto


rust-rdkafka

https://github.com/fede1024/rust-rdkafka

kafaka队列功能 python3

https://github.com/joowani/kq


云原生 cicd argo

https://argoproj.github.io/


## 因子平台

我们上线的，当是稳定的策略，策略盈利是我们的最终目标。实现这个目标，有两个必要条件：

1. 提高预测能力：需要构建更优秀的模型，深度挖掘市场规律

2. 降低风险：设计风控规则，并严格执行止盈止损、仓位控制

3. 风险控制这一块，和实盘交易高度相关后面在考虑讨论和研究，我们将专注于量化模型，并将从机器学习的角度展开一个新的量化投研框架

### 流程
我们曾经提过，所谓的量化交易策略，本质上就是一个机器学习模型，(量化开发之向量化回测框架)

    y=f(X;λ)
这个模型通过历史数据 X,y，寻找历史规律 f，从而对未来的市场走势进行预测。既然同属于机器学习模型，那研究的流程也离不开这么几个步骤：

1. 定义问题：分析需求，定好目标函数
2. 整合数据：收集数据并进行探索性分析与预处理
3. 特征工程：数据清洗、特征衍生、特征筛选
4. 模型训练：算法选择、调参、融合
5. 模型评估：多维度评估样本内外表现
6. 项目交付：测试、模拟、生产

![6fb421ef400579aa67b319c87e1c93ed.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p355)

![0336427e10ee6acd152a0bef874436cb.png](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p367)



#### 1. 定义问题：

###### a. 确认需求
首先，是确定好我们的策略需求，而对量化交易而言，需求可概括为两类——择股 & 择时

*     择股，一般是从市场众多资产中挑选优质组合并长期持有，偏向价值投资，属于低频交易策略。方法上，则是定期对股票进行评估打分，分高者进入组合，概念上与回归模型无二。
*     择时对应的则更像分类模型，判断某个时刻是买、卖还是持仓不动，应用上偏中短线，如波段策略、配对交易、统计套利等。

#### b. 明确目标
当确定好需求后，下一步就是量化对应的目标函数

*     对于择股模型，可以选择股票未来N天的表现当做目标 y，从而让模型 f 来拟合出其与因子数据 X 之间的关系
*     若更关注收益，可以纯用收益率作为目标
*     平衡点的，可以使用夏普比率
*     若风险十分重视，则可以把波动率或最大回撤作为惩罚项加入目标函数
*     同样的，对于择时模型，则可以对未来 N天的收益率进行转化而形成 y：
*     下一阶段资产上涨，标记为1，代表建议买入
*     下一阶段资产下跌，标记为-1，代表建议卖出
*     下一阶段资产波动幅度较小标记为0，代表建议持仓不动，避免频繁交易带来的高手续费

#### 2. 数据整合

定义好 y 后，就该去寻找 X，即我们的核心数据集。这一阶段，主要有4个环节

##### a. 数据收集
巧妇难为无米之炊，数据是模型的基础。一般来说有3种方式收集数据：

专门的数据提供商的接口，如Wind、Bloomberg
交易所API

其他第三方渠道，如tushare、论坛、财经网站数据等

方式又分:

    1. 定时任务数据收集入库
    2. 实时任务收集入库


##### b. 数据清洗
这里我们做最基础的两步预处理

*     缺失值处理：大部分模型，如果有缺失值 nan 的话，都是无法运行起来的，缺失值处理是必要环节。这里研究员可以凭借经验，选择填充亦或是舍弃
*     极值处理：如果说缺失值让模型无法运行，那么极值很可能让模型失效，除非是异常检测模型，否则，建议对数据中的极值进行一定的处理，避免极值对模型产生较大的扰动

##### c. 探索性分析
这个步骤我们确认数据质量、了解其数据分布及特点，并初步检验数据因子的效果。检验方式：

1. 对于单资产的时间序列，可以使用WOE(评分卡模型)，可以把连续型的特征分箱离散化，检验效果
3. 对于多资产的时间序列，则可以使用Alphalens(分组回测方式)，和WOE的思想很相似，核心也是根据因子取值分组，检验各特征组与最终的目标之间的相关性

##### d. 数据集分割
模型研究过程中不可避免地要有 训练集、验证集、测试集。对于时间序列模型而言，经典的分割方式有两种

三段式：训练集、验证集、测试集顺序地分割全数据集
滚动式：将数据集滚动地按顺序分成多份，每份里都是一个三段式地切分

#### 3. 特征工程

一般来说，基础数据之中并不会直接显露出很多信息，要想让模型从中挖掘中这些深层的隐藏信息并不容易。直接训练的话，往往费时费力还无法收敛到一个理想的水平。

![bd090ac19c4b952802cff03381d6bb72.jpeg](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p356)

特征工程的加入，可以很好的解决这个问题，因为它直接将人们多年的经验融入到模型之中，让模型可以站在“巨人的肩膀”上继续学习，加速其优化速度。同样的，特征工程也有4个重要的概念：


##### a. 特征生成
特征生成，也就是我们所常说的因子的生产

这里我们对原始的单时序特征进行处理

    1.     简单如移动平均(MA)、同比环比(DIFF)，以及各类技术指标（BOLLING、RSI）
    2.     统计型处理如核函数、频谱特征
    3.     离散化、归一化、标准化

这块可以有几个很nice的库值得被推荐：talib、tsfresh、featuretools、categorical-encoding

##### b. 特征组合
特征之间可能具有很高的相关性，例如GDP可能看不出和指数的相关性，M2单独看也没有，但是两者组合成 GDP-M2 可能就很不一样了。特征的组合方式也很多样：

*     双特征式的加减乘除
*     统计式地抽取新特征，如PCA前N个主成分

##### c. 特征学习
很多时候，特征可能不仅是原始的数据，而是其他的模型数据，如Kaggle中最受欢迎的模型之一 GBDT+LR，就是讲GBDT的中间产物作为新的特征提供给LR模型进行训练


##### d. 特征筛选
特征的衍生方式数以千计，但并不是说特征越多越好，过多的特征反而会带来一系列副作用。维数灾难让计算量骤增，甚至带来不必要的过拟合。剔除冗余特征才能抽丝剥茧，降低模型学习的难度。筛选的方法也很多，按西瓜书总结的三种：

1.     过滤式选择
2.     包裹式选择
3.     嵌入式选择

#### 4. 模型训练

终于到了挖矿环节了——模型训练

模型训练可以很简单，一个LR模型判断买卖，也可以很复杂，一个巨大的神经网络集成模型。复杂度从低到高，可以分为以下4块：

**平台应该支持：大规模分布式训练场景**

##### a. 算法选择
用什么方式来挖矿呢，可以是铁锹，也可以是挖掘机

* 传统CTA：前人的智慧很珍贵，我们完全可以将CTA视为一个简单的分类模型。CTA一般适用于中低频的价量数据

* 机器学习模型：当模型涉及到了大量财务、经济数据，统计学习就该登场了，Lasso、LR、SVM、GBDT等都是优良的备选模型

* 深度神经网络：而当特征数进一步扩展，还加入了大量的高频tick数据，这么大的数据量，这么高的复杂度，或许该请出AlphaGo，试试神经网络这个魔法黑盒吧

##### b.参数调优（超参搜索）
这是老话题了，好的模型通常都依赖于好的参数选择，这也是不少人都自嘲自己是炼金术师的原因。
![a5c820d0c1852b650853c788784a6127.jpeg](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p357)

但匠人应该熟悉模型，有经验的研究员能设置精准的参数空间，从而让模型快速收敛。

学习器模型中一般有两类参数，一类是可以从数据中学习估计得到，我们称为参数（Parameter）。还有一类参数时无法从数据中估计，只能靠人的经验进行设计指定，我们称为超参数（Hyper parameter）。超参数是在开始学习过程之前设置值的参数。相反，其他参数的值通过训练得出。
参数空间的搜索一般由以下几个部分构成：

*     一个estimator(回归器 or 分类器)
*     一个参数空间
*     一个搜索或采样方法来获得候选参数集合
*     一个交叉验证机制
*     一个评分函数

##### c. 算法融合
一千个读者，一千个哈姆雷特。我们训练得到的模型，只是对历史规律的一个估计。是估计，就一定有误差（偏差+方差+噪声）。噪声不可避免，方差源于数据，而偏差则可以通过多模型组合进行处理，这便是集成学习的价值所在

* 最简单的集成就是voting，多个模型投票，取其胜者

* 然后是bagging、boosting

* 还有stacking，可以把一组模型输出的结果当做新的数据，喂给下一层的meta模型

##### d. 结构学习
既然我们能把模型集成在一块，那么新的问题就来了，
1. 该选哪几个模型进行集成？
2. meta模型选什么？
3. 要不要做多层的集成学习？换句话说组合方式也可以视为一种参数，要纳入到我们的优化范围中。其实到这一步，我们已经十分接近“AutoML”

#### 5. 模型评估
任何模型都有输出，拿量化分类模型来说，模型的输出，则可以视为交易信号。

模型的评价，本质上是对模型的预测输出和真实值得差异情况评估。对于量化模型而言，其输出一般意义上讲就是对下一阶段市场的判断，其评估的重点就在于判断的准确度高低。这里我们有3种方式以供选择

1. 经典的 ∣y_true​−y_predict​∣，用预测值和真实值之间的“距离”衡量准确度（推荐直接使用sklearn.metrics）

2. 量化模型属于经典的时序模型，为了保存其时序维度的信息，我们可以对模型产生的净值曲线进行分析，得到夏普、最大回撤等指标来衡量模型优劣（推荐使用empyrical）

3. 当然，可以再深层次点，用多个绩效指标复合，甚至可以对不同时期给予不同的权重，例如重视风险的话，可以在股灾期间加大评分权重


#### 6. 模型上线（项目交付）
经过层层考验，我们终于获得了一个测试效果良好的模型，我们的研究员小伙伴当是兴奋又疲惫的。当然，感谢我们的量化中台，交付时，一切都变得很简单快捷，研究员只需将代码提交到版本库，我们的系统就能自动进行后续的工作，包括时间驱动回测、模拟交易、以及实盘交易使用


1. 感谢mlflow：模型训练一次不易，需要把训练结果、参数、报告及时持久化下来
2. 感谢airflow（后续使用argo技术）：模型大了，难免涉及分布式，原始数据可能也来源于其他模型的输出，灵活的调度系统少不了

3. Tekton CI-CD：市场变化、知识迭代都快，模型也需要频繁更新，需要一个持续集成系统来整合并自动化所有的交付流程

4. ray分布式计算框架来应对大规模训练场景（支持GPU）

5. 还有我们的AI-PASS平台基座，是一切的基础

### 特征工程（因子）

参考资料

http://wiki.bigquant.ai/pages/viewpage.action?pageId=338666850

http://wiki.bigquant.ai/pages/viewpage.action?pageId=137691428

自动特征因子
https://zhuanlan.zhihu.com/p/67832773

推荐
https://www.cnblogs.com/wkang/p/10380500.html

### 因子分类

1. 量价因子

2. 财务因子

3. 资金流因子

4. 技术分析因子

5. 估值因子

6. 股东因子

7. 波动因子

8. 动量因子

9. 流动性因子

10. 波动因子

11. 高频因子



### 落地实现

确定了以上步骤，我们对量化研究的轮廓就更清晰了一层。但看到这里，有小伙伴可能要后怕，这工作量也太大了，能整合的起来吗？

![d694593ffb974f131fc0ff7bacd8044a.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p358)


感谢我们最初选择基于sklearn开发的量化回测框架(还有其他的成熟方案)，使我们能全面拥抱sklearn-family的生态，可直接复用大量优秀的框架。

框架可参考 awesome-machine-learning
故而按照sklearn的习惯，我们将以上各个步骤抽象，并按照其接口标准进行对象化

*     数据整合：Generator
*     特征工程：Transformer
*     模型训练：Estimator
*     模型评估：Scorer

每个对象都将是无状态式的，像管道一样（暂且称之为管道对象）。而将这些步骤串联起来的方式也颇为简洁，用一个Sequential对象即可（浓浓的Keras味道）：
```bash
seq = Sequential()
seq.add_transformer([transformer1, transformer2])
seq.add_estimator([estimator1, estimator2])
seq.add_estimator(estimator3, meta=True)
seq.fit(train_generator)
seq.predict(test_generator)
```
如此，一个自动化的量化研究框架雏形就出来了。

用图表来表示的话，就像一个神经网络一样，每个管道对象对应的就是神经网络上的一个节点。我们只需要编写好脚本、确定要训练整合的对象及其参数空间，后续的网络结构、参数调优都将由系统自动为我们完成，概念上和NAS（Network Architecture Search）十分相似。

当然，这方面我们同样也不需要造轮子，TPOT、AutoSkleran、AutoKeras、MLens等都提供了很丰富的automl功能，非常值得在量化研究的业务之中复用。


![0df3186c2e29c3e58267dc3f02804889.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p359)


至此，总结一下我们的规划。我们从基本的机器学习模型研究流程讲起，并将其映射到量化研究上，以管道化的方式设计并连接 数据整合、特征工程、模型训练、模型评估等模块，以AutoML的方式进行模型研究。这套模式虽有一定的复杂性，但也裨益颇多：

无需在基础设施的架构耗费时间，开源社区已提供了众多优秀框架
研究员可更关注于业务，只需根据模型需求，调用现成的管道对象；若有必要，再设计新的generator、transformer、estimator或scorer即可
新设计的这些管道对象也具有极高的复用性，可复用于其他模型、亦可分享到团队里
接口统一，便于团队内部的协作建模，可以你设计transformer，我负责estimator，并行工作，加速上线效率
流程高度自动化，调参、组合均由系统完成，避免了人肉调参的低效与主观性，提高模型研究的产出效率


## 数据中台

数据中台构建，中台的最重要的核心技术之一：ETL体系。



### 一、前言

数据中台对于量化投研业务的意义，核心目标是提供一套 覆盖更广、时效更快、数值更准 并 更易使用的数据服务，并为之设计了一套实践蓝图，现在，该是时候把这个蓝图的细节再完善完善了。

通常，数据中台必须有一套中台技术体系支撑，要至少具备 数据汇聚、数据开发、数据可视化 三大功能。

* 数据汇聚：让散落在企业内外部，不同团队或不同项目中的数据资源，汇聚一起互联互通起来，在物理层面打破数据孤岛问题

* 数据开发：汇总好的数据通常没有经过深度加工、还是按原始状态堆在一起，如此业务上使用依旧不便，因此我们会进一步进行加工提炼，以增强数据的使用价值。

* 数据可视化：为了让数据可用易用，需要提供便捷的数据服务可视化能力，帮助业务人员快速了解数据、使用数据、从而产出业务价值。
* 本文的重心在前二者，这两个过程都需要提供适合的数据处理工具，保证数据能被快速采集到统一数据池中并衍生出目标数据集。

**毕竟，数据就像那深埋地下的矿藏资源一样，如果没有配套设施进行采集、存储、加工等步骤，那也只是一堆石头罢了。**

我们将基于量化投研的业务特点，以一套定制化的ETL框架为核心，阐述我们的数据中台实际落地过程中的实现方案。

### 二、何为 ETL

ETL，是英文Extract-Transform-Load的缩写，用来描述将数据从来源端经过抽取（extract）、转换（transform）、加载（load）至目的端的过程。

其目的是将企业中的分散、零乱、标准不统一的数据整合到一起，为企业的决策提供分析依据。

广义上来说，数据的下载、同步整合、指标的衍生计算都算是ETL，故可以想象，ETL在数据体系建设中，将被频繁使用。

从字面上看，ETL似乎只有 E、T、L 三个步骤，但是在实际生产中，并没有这么简单，我们细细展开。

![d0914e0fd1408938fb33013044a7de9f.png](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p360)

首先，一个基础的数据处理任务需要 E、T、L

E: 从数据源中抽取目标数据
T: 对数据进行清洗、加工
L: 将处理好的数据写入到目标源中

并且，会配置相应的检查步骤，包括：

1. 会前置一个任务触发步骤，检查当下是否合适执行任务
2. 会后置一个数据质量校验，检查写入的数据是否满足规则要求


然后，每次任务执行之后，需要将执行情况记录下来，如数据写入量、任务执行时间等，通常会使用日志及数据库来记录

再然后，实际生产中的数据集，通常需要多次任务执行。如

1. 初始化时，需要进行回填历史数据
2. 遇到大数据量的任务时，需要切分为若干个小任务
3. 特别的，任务失败时，需要重试

对任务的拆解，也遵循分而治之的原则，目标是降低任务的复杂度，后续开发过程中也能减轻用户的心智负担。如对于金融投研这个典型的时序数据业务，我们主要会在时间和标的两个维度上对任务进行拆分执行。

最后，配置任务执行情况检查。因为任务进行拆分处理，故上文提及的单任务的数据质量检查，并不能反馈全局的信息，如时间完整性、标的数量完整性、以及任务本身的触发数量水平。

要特别重视数据的质量，garbage in garbage out，这块再严谨都不过分。

#### 三、架构设计

介绍完基本的ETL流程，可以发现实际生产中数据处理步骤远远不止3个。

考虑到数仓建设过程中有大量此类任务，如果没有为这些ETL操作提供一套标准的框架，那么就很容易出现以下问题：

1. 开发效率低下。每个ETL任务中，尽管业务逻辑不同，但代码操作中能沉淀下不少相通的步骤，除了基础的ETL操作，还有其他辅助环节如任务回填、数据库读写接口、数据验证算子等，自起炉灶耗时耗力。

2. 部署流程不畅。工程师开发完成了ETL代码更新以后，还需要到不同平台配置信息，如调度平台配置任务，元数据平台配置文档、上下游信息等，内容琐碎且重复。

我们希望的是：ETL任务只要选取相关套件、并填写关键的业务操作逻辑，剩下的交给系统自动完成即可。

经过一番调研发现，ETL这块开源社区提供的都是较为简单的框架，如 kettle、pyetl。而我们需要的框架，不仅需要覆盖以上提及的环节，还要需要针对投研领域多时序数据的特点，定制相关的功能，同时能桥接大量我们自选的其他系统，如数据系统、调度系统、报表系统、元数据系统等。

故而这块我们选择灵活性更好、掌握性更强、迭代速度更快的自研框架。

这里我们将采用流水线式架构设计这套框架。

![ddcdab6dec303c8f79cd5e69756807bd.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p361)

对于每个ETL任务，我们都抽象成一个数据流处理器 DataHandler，由多个子组件组合生成。按照执行逻辑，组件包含如下：

1. Trigger: 检查当前时间是否是可以触发任务运行。通常用于当前是否是交易日、交易时间。

2. TaskGenerator：生成子任务。当天可以执行任务时，可以执行多个任务实例，如回填N天的任务；或者针对每一个标的生成一个任务实例。故而有两个层次的任务生成器：DateGenerator + CodeGenerator

3. DateGenerator：用来生成具体的任务执行日期列表，如工作日、交易日、财报发布日等

4. CodeGenerator：用来生成需要进行数据处理的标的列表，如全A股股票列表

5. DataReader：读取数据源数据

6. DataProcessor：数据清洗与衍生，承载具体业务逻辑

7. DataValidator：数据校验，如数据缺失检查、类型检查、重复检查等

8. PersistHelper：数据持久化，将数据批量写入数据库

9. TaskReporter：记录任务的执行结果元数据

10. TaskChecker：任务检查与全局数据质量校验


其中，每个组件都是无状态化的设计。无状态组件具有非常强的封装性，执行后无残留影响，减少编码人员的心智负担，从而降低耦合的可能，使各个组件能够更加灵活地组合使用。最直接的例子就是scikit-learn中的Pipeline类，数据就像管道中的水，从一个组件流向下一个组件。

同时，我们利用Python的多继承特性，我们给特定场景的任务引入Mixin类以支持相应特性：

1. IdentifierMixin：逐个标的执行目标任务

2. TsPointMixin：逐个时间点执行目标任务

3. TsRangeMixin：更行上次更新日期至今的数据

4. IdentifierTsRangeMixin：每个标的都自动按照时间区间执行任务

5. PartitionMixin：分区特性，执行的任务将只关联到特定分区表数据

于是，当我们要实现一个每天下载行情数据的下载器时，可以如此设计代码：

```bash
class PriceDownloader(TsPointMixin, DataHandler):
    default_start_date = datetime.datetime(2005, 1, 1)  # 默认从2005年开始回填下载
    table = 'ods_mkt_stock_cn_daily_price'  # 目标数据库表
    data_reader = ApiPriceReader(price_fields=price_fields, freq='d', adj=True)  # 调用历史行情接口读取特定字段数据
    date_generator = TradingDateGenerator(date_type='SSE', start_offset=1, end_offset=0, freq='d')  # 生成 数据最后一天至今的所有交易日 的日期列表，用以下载数据
    code_generator = SecurityCodeGenerator(sec_type='stock_cn', exclude_exist=True)  # 可以生成交易当天的A股所有上市股票代码
    chunk_size = 1000  # 每次下载时，会将股票分批，每次下载1000个股票的单日行情数据，避免超过接口数据限制
    task_checkers = [
        DateIntegrityChecker(),  # 检查是否每一个交易日都有数据
        CodeIntegrityChecker(),  # 检查是否每一个标的都有数据
    ]
```


这里我们使用了继承的方式来实现每个具体任务的业务逻辑，有点类似于Django中的Model类，每个Model类都正好映射到数据库中的一个表，正如我们每个Downloader对应一个目标表

不过编程的最佳实践里一直有这样的一句话 慎用继承，优先使用组合

其中有个原因是 在继承结构中，子类与父类之间是紧耦合的，如果基类的实现发生改变，那么派生类的接口实现也将随之改变。这样就导致了子类行为的不可预知性，同时大量的新增子类也将增加系统的复杂性。

若我们纯采取继承设计时，也会有这劣势。故权衡以后，我们约定子类衍生设计时，不覆盖父类的接口方法，而只替代父类的类成员变量，来生成一个新的任务类，同时继承的层级尽量限制在两层内。让框架的形式上介于继承与组合之间。

组合更安全，更简单，更灵活，更高效。而继承则可以精简代码、提高复用性与开发效率。

这样的设计很适合彼此默契的小团队配置使用，框架能在可维护的情况下提供最佳的灵活性和开发体验

### 四、框架组件开发

对整体框架有了大概的认知后，下一步就是细节上的组件设计了

#### 1. Trigger 检查任务是否是执行时机

```bash
class TaskTrigger(object):
    def check_status(self, dt) -> bool:
        pass
```
通常任务都是有触发时机的，或者在某个给定时间执行，或者在满足某些前置条件时执行。

通常调度系统都很擅长处理任务间的依赖条件触发，这块我们可以不必操太多心。如在调度系统 Airflow(Argo) 中，我们可以使用 Sensor 算子大部分时候我们可以直接使用。

如 可以选择依赖前置的数据下载任务、或者数据检查任务的状态，当且仅当前者状态为成功时，方才触发执行。

其他涉及业务逻辑的触发逻辑，则可以通通交给 Trigger 进行处理，如检查当前是否是某个交易所的交易日期、当天是否有新增新上市的股票债券。

#### 2. DateGenerator 生成任务执行日期列表
```bash
class TaskDateGenerator(object):
    def get_target_task_dates(self, handler: 'HandlerInterface', dt, code=None) -> list:
        pass

```

投研领域大部分数据都是时序数据，操作时序数据时需要非常注意增量化处理。

当没有DateGenerator时，我们需要依赖调度系统完成回填工作（back-fill），如 **Airflow**(Argo) 默认会将把新增的任务，从 start_date 起按照 schedule_interval 为间隔回填到最近一天。

但是这种回填在实际生产中也存在明显劣势，那便是性能损耗。

基本所有的调度系统在执行单个原子任务时，都要新起一个进程、甚至一整个容器，避免任务错误导致调度系统的主进程崩溃。

但进程是很重的，会导致实际上十几毫秒可执行完成的任务，要耗费数秒进行进程重启与环境初始化工作，同时进程中针对慢查询的缓存等也会一并消失，下次启动时又要耗费大量时间进行IO。

另外再灵活性上来说，并不是所有的业务都是按照自然日进行任务执行的，如下载每个季度的财务数据下载是季度日、周频信号计算是每周最后一个交易日。 如果依然Airflow(Argo)，那么就必须要结合 Trigger 进行日期跳过。而这样就又必须起一堆进程执行任务，回到了第一个痛点上。

故而引入DateGenerator，可以灵活地直接在单个进程里生成目标任务日期列表、执行任务，提高任务运行效率。

```bash
class TaskDateGenerator(object):
    def get_target_task_dates(self, handler: 'HandlerInterface', dt, code=None) -> list:
        pass
```

投研领域大部分数据都是时序数据(时序数据库存储)，操作时序数据时需要非常注意增量化处理。

当没有DateGenerator时，我们需要依赖调度系统完成回填工作（back-fill），如 Airflow(argo) 默认会将把新增的任务，从 start_date 起按照 schedule_interval 为间隔回填到最近一天。

但是这种回填在实际生产中也存在明显劣势，那便是性能损耗。

基本所有的调度系统在执行单个原子任务时，都要新起一个进程、甚至一整个容器，避免任务错误导致调度系统的主进程崩溃。

但进程是很重的，会导致实际上十几毫秒可执行完成的任务，要耗费数秒进行进程重启与环境初始化工作，同时进程中针对慢查询的缓存等也会一并消失，下次启动时又要耗费大量时间进行IO。

另外再灵活性上来说，并不是所有的业务都是按照自然日进行任务执行的，如下载每个季度的财务数据下载是季度日、周频信号计算是每周最后一个交易日。 如果依然Airflow（argo），那么就必须要结合 Trigger 进行日期跳过。而这样就又必须起一堆进程执行任务，回到了第一个痛点上。

故而引入DateGenerator，可以灵活地直接在单个进程里生成目标任务日期列表、执行任务，提高任务运行效率。

常用的DateGenerator子类有：

1. NatureDateGenerator：生成自然日列表
2. TradingDateGenerator：生成交易日列表
3. FuncDateGenerator：生成自定义日期列表

而在实际任务回填过程中，不仅会按照时间进行切分，也会按照标的进行切分。这就引出了我们下个一个要介绍的组件，CodeGenerator

#### 3. CodeGenerator 生成任务执行标的列表
```bash
class TaskCodeGenerator(object):
    def get_target_identifiers(self, handler: 'HandlerInterface', dt) -> list:
        pass
```
投研领域存在大量 标的 + 时间 为细粒度的数据，标的可以为 股票、基金、指数、板块、宏观指标等。对于这些数据，非常推荐使用 CodeGenerator 生成每期的任务列表，这样

1. 可以切分任务，简化任务处理逻辑，减轻开发时的心智负担

2. 任务数据通常在时序维度要有依赖性，但在标的维度是没有依赖性的，于是我们便可以在标的维度上进行充分的并行化任务处理

3. 对于数据下载任务，由于大部分数据接口都有严格的下载限制，可通过 CodeGenerator 进行数据分批操作，以减少单次数据请求量


最常用的是 SecurityCodeGenerator，相关参数为
```bash
sec_type：指定标的集合，如 stock_cn, stock_hk, index_cn
exclude_exist: 是否排除当前日期下已下载到的目标标的，主要是用来避免重复数据下载的
```

基于 DateGenerator + CodeGenerator 两个组件，我们即可生成出目标子任务集，其中的每个任务都是不可再分的原子任务

对于每个原子任务，我们会给于3个重要的初始属性

1. identifier: 标的，无标的任务可以是 None，分批型任务可以是 List[str]，更通常的则是 str

2. interval_end_date: 原子任务跨度结束日期，通常是 DateGenerator 生成的某个任务日期

3. interval_start_date: 原子任务跨度开始日期，如，大部分时序任务的开始日期，是给定 identifier 条件下的最后一条数据再向后偏移一个时间单位的日期

![60d7b4c7046a8bc49f39768ff10fc7cb.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p362)
原子任务得到关键的上下文变量后，便可以开始具体任务执行

#### 4. DataReader 进行数据载入

数据下载(arrow对象？)
```bash
class TaskDataReader(object):
  def fetch_data(self, identifier, start_date, end_date) -> 'pd.DataFrame':
    pass
```


数据任务的第一步，便是数据载入。

数据通常是由外部系统载入，故需要注意以下4个事项：

1. 重试：数据载入涉及到IO操作和外部数据访问，必然会遇到不稳定问题，因而须在该步骤中添加 失败重试器。这块可以直接使用库 retrying，基本包含所以重试的逻辑。

2. 限流：同时，部分数据接口和网站会有访问限制，限制单位时间内的数据访问次数，这块可以使用令牌桶算法进行处理。

3. 标识符切换：通常外部数据的标识符和内部数据的会有一定的差异，如A股代码表示法，可以是 600000.SH，也可能是 600000.XSHG，故而需要酌情添加一个 identifier_proxy 组件进行标识转换

4. websocket支持


而落地到量化业务，由于大部分数据都是时序属性的，不少处理方式要求要进行有前置的缓冲数据，如 rolling 及基于其上的技术指标计算等，都需要读取历史以前一段时间的数据。故而需要考虑在模型处理时，添加一个上下文属性变量充当前置的时间天数。

```bash
class DataHandler(object):
    ...
    def _execute_task_impl(self, identifier, dt):
        # 执行原子任务

        # 确定任务时间区间
        start_date = self._get_interval_start(identifier, dt)
        end_date = self._get_interval_end(identifier, dt)

        if start_date > end_date:
            return

        # 部分任务会需要有前置缓冲量的时间区间，如 rolling、同比环比 等场景
        reader_start_date = self._get_reader_buffer_start(identifier, start_date)

        # 访问外部系统时，会有不同的标识符，需要按需进行切换
        identifier_proxy = self._get_reader_identifier_proxy(identifier)

        # 实际的数据读取过程
        data = self.data_reader.fetch_data(identifier_proxy, reader_start_date, end_date)

        if data is not None and len(data):
            # 管道式数据处理
            if len(data):
                data = self._process_data_impl(data)

            # 验证数据质量
            if len(data):
                self._validate_data_impl(data)

            # 持久化数据
            if len(data):
                self._persist_data_impl(data)
```
特别的，这里我们并没有直接让DataHandler 直接去引用各个组件进行数据处理，而是使用了函数方法进行封装。此举是为了方便引用 Mixin 类，提高框架的灵活性，我们后面涉及时会再展开。

于是，在设计DataReader时我们便可以将注意力集中在数据载入逻辑上，让载入逻辑尽可能精简。

其接口fetch_data(self, identifier, start_date, end_date)中的三个参数与原子任务重的上下文属性基本一致，

如对于ApiPriceReader而言，该接口返回 对应标的identifier在时间区间【start_date，end_date】范围内的行情数据

其他常见的DataReader还有

DbDataReader：从特定数据库中的数据表中读取数据
ApiDataReader：从第三方数据api中获取各类数据
FileDataReader：从数据文件中读取文件
QueueDataReader：从队列中读取数据

#### 5. DataProcessor 数据清洗与衍生，承载具体业务逻辑
```bash
class DataProcessor(object):
    def transform(self, df):
        pass
```
DataProcessor会将载入的数据进行业务加工处理工作。

DataProcessor将采用典型的管道类型设计，就像sklearn.pipeline中的由多个Transformer构成的Pipeline一样，DataHandler提供data_processors属性来定义处理流水线


管道中的数据通常都是DataFrame类型，常用的加工动作有

1. 字段映射重命名
2. 字段类型转换
3. 新增标记列
4. 字段数值计算衍生、聚合

#### 6. DataValidator 数据校验

```bash
class DataValidator(object):
    def validate_data(self, df) -> bool:
        pass
```

在数据存储到数据库前后，都需要进行相应的数据质量检查工作。

数据通常是由原子任务执行单次操纵，一批一批地插入到数据库之中，因而在完全插入完成前，一般只进行数据本身的质量检查；待到任务完全执行完成后，再行进相关的完整性检查。

DataValidator负责的便是数据入库前检查，通常包括数据缺失、重复、类型错误、取值范围等。

当然，如果使用的是关系型数据库作为数仓后端，其实这类工作也可以交给数据库进行处理。数据缺失用NOT NULL检查，重复使用Unique INDEX检查，类型使用DDL定义即可。

若是使用MongoDB等非关系型数据库(clickhouse)，则以上的步骤不可缺少。

#### 7. PersistHelper 数据的持久化
```bash
class TaskPersistHelper(object):
    def persist_df(self, handler, table, df):
        pass

```

TaskPersistHelper顾名思义，是将数据存储到机器上的组件。可以是文件形式、缓存形式，当然最常见的当属数据库存储。

通常TaskPersistHelper将调用我们的db_conn对象，将数据存储到数据库中对应的表上面。

特别的，对于数据库连接对象db_conn，尽量考虑采取DataMapper的形式，将数据从内存里的DataFrame映射到数据库的表结构上，DataFrame的列名即对应表格的字段名。

这样设计可以隔离具体实现，既能兼容不同数据库类型时，也可保证后期如果对数据库进行迁移，程序能够较为顺利地切换。

最简单的实现其实就一行代码，即df.to_sql，但是考虑到完整的数据操作至少要有增删改查4个，比较建议为之单独设计一套数据库调用方案。

回到主题，常见的TaskPersistHelper包括

StandardPersistHelper：标准的插入所有数据
OverwritePersistHelper：根据待插入数据的标的或时间范围，先删除数据库中的原有数据，再插入现有数据
UpdatePersistHelper：根据给定的键值更新数据库记录

#### 8. TaskReporter 记录任务的执行结果元数据

```bash
class TaskReporter(object):
    def report_result(self, handler, execution: 'ExecutionStatistic'):
        pass
```

一般调度系统都有较为完整的任务执行元数据记录，其关注点在于任务自身的状态。故而我们就不再重复造轮子，我们的TaskReporter将会把注意力转移到任务操作的数据本身上。

TaskReporter将记录每个原子任务生成的：

1. 任务标的数量
2. 任务跨度开始日期与结束日期
3. 任务结果数据的标的数量
4. 任务结果数据的开始日期与结束日期，以及日期数量
5. 任务结果数据的行数与字段数
6. 任务执行的开始时间、结束时间，以及总耗时
7. 并在所有原子任务完成后，再对以上数据进行聚合，获得当天的任务执行结果情况。

有了这些元数据，我们便能更好地了解ETL任务的实际执行情况，并为其配备更周全的质控操作。

#### 9. TaskChecker 任务检查与全局数据质量校验
```bash
class TaskChecker(object):
    def check_result(self, handler, dt):
        pass

```
毫无疑问，ETL框架的最终追求都是高质量数据。然而，在实际实施过程中，难免会因为程序开发遗漏或是外部接口不稳定等，导致数据未达预期。

这个时候，发现问题并进行修复就显得额外重要，而且越早发现越好，毕竟数仓存在大量的数据依赖，源头的数据错误，后续一系列的衍生数据就难逃被污染的结局。

如前所述，部分数据数据校验工作已经由DataValidator组件完成，这里的TaskChecker将更关注数据完整性以及时效性

**首先，要关注完整性，数据完整性是数据质量最基础的保障。**

**完整性是指数据的记录和信息是否完整，是否存在记录的缺失或重要字段信息的缺失，两者都会造成统计结果不准确。**

在投研数据中，最重要的就是检查日期、以及标的的完整性。如检查每个发布日都有数据，检查单个日期的标的数量都达到当天的上市公司数量。

在具体实现上，我们可以结合任务本身CodeGenerator以及DateGenerator完成完整性检查。

比如，下载时序数据时，我们通常会采用TradingDateGenerator，当我们检查每个交易日是否都有数据时，如果数据下载完全，那么新生成的目标任务日期列表就是空，以此变可以判定日期列表是完整的。（包装成DateIntegrityChecker）

同理，当我们检查某个交易日（通常是最新交易日）的目标标的数据都下载完全时，也可以用SecurityCodeGenerator(exclude_exist=True)得到最最新标的列表是否为空，来判断标的数据是否下载完全。（包装成CodeIntegrityChecker）

另外，我们也会检查数据条数。此时我们会根据TaskReporter记录的任务元数据，观察每一期的数据插入条数，如果每天约为10万条记录，某天却突然下降了1万条，那大概率也是出现了记录缺失的情况。

在确保数据完整下载后，便是时效性检查，毕竟只有数据及时入库，业务方才能顺利调用。

时效性上常见的有以下两类检查：

1. TimeEfficiencyChecker 是否关键时间节点前完成：即给定一个时间，在这个时点前任务是否顺利完成，保证不影响后续业务调用

2. DurationChecker 执行时间是否叫之前大幅上升：如果往期10分钟都能完整的任务，此次要花费半小时，那么有必要检查是否某处出现了代码纰漏或者资源挤占，需要及时进行应对调整

#### 10. 其他属性

以上9个组件基本完成了一个ETL任务重的所有核心操作

还有几个比较重要的类属性，我们分别介绍一下
```bash
exec_mode = 'date'  # 可选值 date, code
cond = None
param_grid = {}
group = ''
priority = 9
db_id = 'local'  # settings.get('db_conn_id')
```

1. 首先是exec_mode。每次执行任务时，我们会根据日期列表和标的列表生成一堆的原子任务，exec_mode帮我们确定我们是先逐个标的完成任务执行，还是逐个日期执行。
2. 之后是cond，本意为condition，对应数据表的具体分区分块。如上文提到的PriceDownloader可以既下载日级别，也可以下载月级别的数据。那么日级别对应的cond变量可以为freq='daily'，直接对应到数据表的分区。如此后续的增删改除都将基于该分区进行操作。
3. 然后是param_grid。通常一个具体的数据下载器，可以根据参数不同而执行不同的业务逻辑。我们这里借鉴sklearn.base.BaseEstimator的模式，让初始化参数均为显性参数（非*arg或**kwargs），且与实例属性一一对应。这样我们便可以基于sklearn的ParameterGrid对象，快速为任务类生成所有的任务实例并执行。
4. priority和group。这两个属性一起，配置了任务的默认归属组以及在其组内的优先执行等级。
1. 最后是db_id。这个属性声明了目标表所在的数据库ID，通常在开发时，对接测试环境数据库，部署时再切换到生产环境数据库（利用dynaconf完成自动切换）。




### 11. Mixin类
Mixin类本质上是利用Python的多继承特性，基于面向对象的继承方式，帮助拓展子类的功能，而不影响子类的主要功能。
![6f8305497d83957634ff178ec682a8eb.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p363)

以IdentifierMixin为例，这个嵌入类的主要应用场景，是逐个标的执行ETL操作的任务类，代码如下
```bash
class IdentifierMixin(object):
    batch_chunk = 1  # 逐个标的进行
    exec_mode = 'code'  # 优先按照每个标的执行完对应的所有日期任务，再执行下一个标的

    @property
    def cond(self):
        cond = [self.db.get_value_filter(self.code_field, self.context.identifier)]
        extra_cond = super(IdentifierMixin, self).cond
        if extra_cond:
            cond.append(extra_cond)
        return cond

    def setup(self):
        super(IdentifierMixin, self).setup()
        self._assert_attribute('code_field')  # 必须提供 code_field 属性
        self._assert_attribute('code_generator')  # 必须提供 code_generator 属性

    def _process_data_impl(self, df):
        df = super(IdentifierMixin, self)._process_data_impl(df)
        if isinstance(df, pd.DataFrame) and (self.code_field not in df.columns):
            df[self.code_field] = self.context.identifier  # 标记该数据块的 code_field 为当次原子任务的 identifier 标的信息
        return df
```
之后的子类 继承IdentifierMixin只是增加了额外一些功能，并不影响自身的主要功能。

比如下载行情时，原PriceDownloader会批量下载数据，若添加IdentifierMixin为基类，则将逐个标的地下载行情数据。

Mixin类把通用的功能抽取出来、精简代码，同时也支持多重继承，如一个具体子类既可以继承IdentifierMixin，也可以同时继承TsRangeMixin。某种程度上，这也是组合的设计模式。

其他Mixin类也大同小异，这里就不再赘述

### 框架应用：投研数仓
ETL框架基本已经设计完毕，是时候投入到实际使用场景中。

这里我们以A股投研数据仓库构建为例

通常，如果团队投研经费有限，不足以购买全量的底层数据库，那么就逃不过自建数据库。自建投研数仓需要

数据下载：从外部数据源或内部其他业务数据库拉取数据。如下载各类行情数据、财务数据、基础信息等
数据衍生：将下载的数据进行整合、清洗、衍生。如财务数据整合、PIT(point-in-time)时间对齐处理、因子数据生产
数据同步：将数仓中的数据同步到对应的业务系统中使用。如导出到行情分红等数据至事件驱动回测系统支持回测、导出数据为本地文件方便研究使用。
显而易见，这几步ETL框架都能提供重要作用，结合好业务经验，就能撬动、搭建好完整的投研数仓

当然，经验通常都比较冗杂，我们这里点几个关键细节，未来有机会再展开细聊。

首先是投研数据下载顺序。

投研数据繁多，需要理清头绪，从日历数据和标的数据这两类元数据展开，再进行全量历史数据下载。推荐顺序如下

1. 交易所日历：大多数据以交易所日历进行下载的，故首先补全该日历数据，如A股交易日历、港股交易日历、美股交易日历等。后续其他任务类的TradingDateGenerator、TradingDateTrigger将依赖于它。

2. 标的代码表：标的代码表包含 标的类型、标的代码、标的名称、上市日期、退市日期等，保证可以拿到任意一天的全部上市股票数据、上市基金数据。后续其他任务类的SecurityCodeGenerator将依赖于它。

3. 财务报告日历：鉴于接口访问量有限，优先查询财务报告的发布日期，仅当该日历
   数据新增时，再下载财务数据。后续其他任务类的ReportCodeGenerator将依赖于它。

4. 行情数据：每日下载仍处于上市状态的标的的泛行情类数据，包括价格、资金流向、市场人气等数据。通常会依赖于TradingDateGenerator与SecurityCodeGenerator

5. 财务数据：根据财务统计日期下载三大报表数据、以及其他常用衍生因子数据。通常会依赖于NatureDateGenerator与ReportCodeGenerator，前者生成季度日期，后者查看该季度日期已发布报告的所有代码

同理，也可以扩展港股、美股、基金、指数、期货和数字货币等数据。

然后是投研数据清洗与衍生，这里提几个重要的衍生数据表


1. 周频、月频交易日日历：基本面因子研究多以周频、月频为主，故基于原生日历表衍生出周频、月频日历表

2. 财务数据发布日期对齐：财务数据通常是当天收盘以后发布，但发布日期会标记为第二天。为了和行情数据的日期标记模式对齐，我们需要把财报上的发布日期前移，与前一个交易日对齐

3. 周频、月频因子财务数据衍生（含PIT处理）：基本面因子研究会要求将财报数据衍生对齐为周、月频数据，具体操作上，每周/月更新数据时，将会根据最邻近的数值进行ffill最新一期数值

4. 收益率数据计算：以当期close计算的收益率并不符合实际交易情况，建议以次期的open或者avg_price进行收益率结算。这块也要注意日期上的对齐操作。

每个细节背后都是实践中踩过的坑，开发数仓需要"get your hands dirty"。


### 生产上线

当一批具体场景的任务类开发完成后，便可以部署到生产环境落地执行。
以下是aiflow的dag任务图（后续迁移到Argo）

![f95ff93ae943548b889770f8a4dfcf36.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p364)
以Airflow为调度系统，结合调度脚本样例ods_download_ms_data.py，进行说明
```bash

import datetime
from airflow.decorators import dag
from skportal.dw import DwAirflowSSHTaskGenerator, dw_notify_failure_callback

default_args = {
    'owner': 'jesse',
    'depend_on_past': False,
    'start_date': datetime.datetime(2020, 1, 1),
    'on_failure_callback': dw_notify_failure_callback(),
    'retries': 1,
}

@dag(
    schedule_interval='0 1 * * 1-5',
    default_args=default_args,
    catchup=False,
    max_active_tasks=4,
    tags=['ods', '晨星数据'],
)
def ods_download_ms_data():
    """原始数据下载，来源：Morning Star"""
    task_gen = DwAirflowSSHTaskGenerator()
    task_gen.read_config_by_db('sklake.ods.ms')
    _ = task_gen.gen_tasks()

d = ods_download_ms_data()
```
我们采取持续交付的部署方案，在推送任务代码到代码仓库后，自动化部署框架会执行


![dfccb7e28950b2181e104f08869e14a7.webp](evernotecid://89DCE52F-0EDB-4C9C-B8D9-33B697CC20C8/appyinxiangcom/18665077/ENResource/p365)

1. 部署任务代码包到给定节点（node）的给定任务环境（env）。当使用k8s执行时，则可以直接打包成单独的容器。

2. 数据任务类的注册。我们暴露一个入口点setup，用来搜索代码包中可部署的所有任务类，将任务元信息打上标记如sklake.ods.ms，再整合记录到数仓的元信息数据库中

3. 任务调度的脚本部署到调度系统。通常会专门为每个项目设置单独的DAG目录，每次部署都采取覆盖更新。

4. 调度系统读取任务元信息，并生成对应的任务运行算子。每个代码包有若干个子模块、每个子模块有若干个任务类、每个任务类将生成若干个任务实例，故一个dag有数十个任务执行也属正常。

5. 算子待到触发时机时，在给定节点的给定任务环境执行。

实际在执行时，我们会让框架暴露一个入口点，如 etl run --class sklake.ods.ms.stock_cn.PriceDownloader --param '{}'

然后设置对应的执行执行算子即可完成部署。当资源有限时，可以以物理机形式部署，指定节点及其虚拟环境执行任务。当资源充足且任务量开始指数级增多时，可以切换到k8s与容器部署的模式进行生产管理。

我们通过

1. 抽象ETL任务所涉及的各个操作
2. 结合量化业务设计ETL框架及其配套组件
3. 基于ETL框架开发投研数仓应用的实践细节
4. 部署ETL任务的持续交付方案

完整介绍了针对量化投研数仓而设计的ETL框架。。

有了这套框架，我们完全有能力快速地搜集与生产数据，一套中小型投研数仓也触手可及。

不过，随着数据的增多，为了用好这些数据，可视化工作也变得愈发重要。毕竟人是视觉动物，直观的图像能让团队在最短时间内理解数据并做出决策，量化投研领域也适用这个道理

1. 投研数据大文件。以同构矩阵模式存储，高速读取；覆盖主流行情、财务数据，以及部分加工的事件、因子数据；05年至今数据，每周更新（行情数据，clickhouse或feather文件）

2. 投研工具包。基于向量化运算量身定制的 回测、绩效分析、可视化框架，以及主流因子代码（如技术因子代码、alpha101等)

3.因子看板。基于开放的因子、事件、策略等数据，展开对应的绩效分析与可视化，每日更新（metabase）