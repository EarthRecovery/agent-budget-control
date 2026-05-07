# Related Papers for Agent Budget Estimation

按 `storyline.md` 的主线，下面把相关文献分成 5 组：`CMDP / safe RL`、`prediction interval / calibration`、`test-time compute`、`LLM self-monitoring / uncertainty / reflection`、`agent benchmark / trajectory environment`。  
以下“摘要”均为基于原始 abstract 的中文整理，不是逐字翻译。共整理 **48 篇**。

## 1. CMDP / Safe RL / Budgeted RL

1. [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) (2017): 提出通用的约束策略优化算法 CPO，在高维连续控制中既优化回报，又在训练过程中近似满足安全约束，并给出相应理论保证。它代表的是“在外生约束下学 policy”的路线，而不是在执行时在线估计剩余 budget。

2. [Reward Constrained Policy Optimization](https://arxiv.org/abs/1805.11074) (2018): 提出多时间尺度的 RCPO，用替代惩罚信号把策略逐步推向满足约束的解，并给出收敛性分析。论文关注的是如何训练出 obey constraint 的策略，而不是 agent 是否知道自己还需要多少成本。

3. [Budgeted Reinforcement Learning in Continuous State Space](https://arxiv.org/abs/1903.01004) (2019): 将 Budgeted MDP 从离散、已知模型场景推广到连续状态和未知动力学场景，提出 budgeted Bellman operator 及其深度 RL 实现。这里的 “budget” 是控制条件的一部分，而非 trajectory 中动态更新的自我估计对象。

4. [Safe Reinforcement Learning in Constrained Markov Decision Processes](https://arxiv.org/abs/2008.06626) (2020): 提出 SNO-MDP，在未知安全约束下先扩张安全区域，再在认证安全区域中优化累积回报，并给出约束满足与近最优性的保证。它很适合用来对照你文中“已有 safe RL 往往默认 constraint model 可学或可得”这点。

5. [Constrained Upper Confidence Reinforcement Learning](https://arxiv.org/abs/2001.09377) (2020): 在已知转移、未知 reward/cost 的 CMDP 里提出 C-UCRL，在学习过程中保持高概率约束满足，同时获得次线性 regret。核心仍是在线控制中的 constraint satisfaction，而不是剩余成本区间估计。

6. [Constrained Variational Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/2201.11927) (2022): 从概率推断角度重写 safe RL，使用 EM 式 E-step/M-step 把问题分为凸优化与监督学习两阶段，以提升稳定性和样本效率。该文强调的是更稳的 constrained optimization，而非元认知式 budget awareness。

7. [Near-optimal Conservative Exploration in Reinforcement Learning under Episode-wise Constraints](https://arxiv.org/abs/2306.06265) (2023): 研究 episode-wise conservative constraint 下的探索，提出 StepMix 和 EpsMix，在不违反每回合阈值的前提下逼近无约束设置的 regret。它强调“全过程不能低于安全底线”，和你的问题一样关注 trajectory，但目标仍是 policy safety，不是剩余预算认知。

8. [Safe Reinforcement Learning for Constrained Markov Decision Processes with Stochastic Stopping Time](https://arxiv.org/abs/2403.15928) (2024): 处理带随机 stopping time 的 CMDP，给出无需环境模型的线性规划式在线算法，并高置信保持安全。该工作进一步说明 safe RL 文献通常围绕停止前 constraint violation，而不是 agent 对 future cost 的主观区间信念。

## 2. Prediction Interval / Calibration / Uncertainty Quantification

1. [High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach](https://arxiv.org/abs/1802.07167) (2018): 论文把“区间要尽量窄，同时覆盖率要够”作为直接目标，推导出无需分布假设的损失，并通过 ensemble 处理模型不确定性。它直接支撑你文中 coverage 与 tightness 的双目标评估语言。

2. [Distribution-Free Predictive Inference For Regression](https://arxiv.org/abs/1604.04173) (2016): 这是 conformal regression 的经典框架，用任意回归器都能构造有限样本有效的 prediction band，并系统比较 full/split/jackknife 等变体。它是你文中“post-hoc calibration baseline”的最基础理论来源。

3. [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263) (2018): 提出一种类似 Platt scaling 的回归校准程序，可把任意回归模型的 credible interval 调整为更符合经验覆盖率。它很适合支撑你对“为什么只做 post-hoc calibration 仍不够”的讨论。

4. [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222) (2019): 将 quantile regression 与 conformal prediction 结合，使区间既保持有限样本覆盖，又能适应 heteroscedasticity，从而比固定宽度区间更紧。它和你文中“区间而非点估计”的 formulation 非常贴近。

5. [Deep Evidential Regression](https://arxiv.org/abs/1910.02600) (2019): 用 evidential distribution 同时学习连续预测与对应证据量，从而联合表示 aleatoric 和 epistemic uncertainty，且推理时不需要采样。它提供了一条“模型内生地输出不确定性”的路线，但对象仍是静态预测器而不是会改变 future cost distribution 的 agent。

6. [Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification](https://arxiv.org/abs/2011.09588) (2020): 论文指出单纯优化 pinball loss 会限制模型类与目标性质，于是提出能在 calibration、sharpness、centered interval 之间灵活权衡的新方法。它可用来支撑你文中对 “tight but unsafe” 与 “wide but calibrated” 区别的强调。

7. [Evaluating and Calibrating Uncertainty Prediction in Regression Tasks](https://arxiv.org/abs/1905.11659) (2019): 这篇文章指出已有回归 calibration 定义可能无法区分“有信息的 uncertainty”和“无信息的 uncertainty”，并提出更直接的 histogram-based 评估与简单缩放校准法。它能帮助你把 failure trajectory 上的“选择性过度自信”表述得更尖锐。

## 3. Adaptive Compute / Test-Time Scaling / Budgeted Inference

1. [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) (2016): 提出 ACT，让 RNN 为不同输入自适应分配不同计算步数，并观察到更难预测的 transition 会消耗更多计算。它是“计算预算应随输入难度动态分配”的早期代表工作。

2. [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](https://arxiv.org/abs/1709.01686) (2017): 通过在网络中加入 side branches，让简单样本提前退出，困难样本继续走更深层，从而降低延迟和能耗。它代表“在固定模型内节省推理预算”的思路，但没有处理剩余预算估计。

3. [SkipNet: Learning Dynamic Routing in Convolutional Networks](https://arxiv.org/abs/1711.09485) (2017): 利用 gating network 按输入跳过部分卷积块，把动态跳层写成 sequential decision problem，并结合监督学习与 RL 优化。该类工作说明“预算感知 compute allocation”已有历史，但目标是算得更省，不是让 agent 知道自己还会花多少。

4. [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](https://arxiv.org/abs/1810.07052) (2018): 系统研究网络 overthinking：模型常在较浅层已可正确预测，却继续计算，甚至后续层把正确结果改成错误结果；作者因此提出内部分类器与 early exit。它和你文中的“trajectory 后段才显出 failure/success 信号”形成一个有趣对照。

5. [Learning to Stop While Learning to Predict](https://arxiv.org/abs/2006.05082) (2020): 论文学习一个可控的 stopping policy，让不同输入在不同深度停止，以避免 over-thinking 或不必要计算。它关注 per-instance optimal depth，和你的工作一样强调“过程中的动态判断”，但对象是网络层数而非 agent trajectory 的剩余成本。

6. [Learning to Skip for Language Modeling](https://arxiv.org/abs/2311.15436) (2023): 在语言模型里提出按 token 动态跳过层或模块的 routing 机制，用不同计算量处理不同复杂度 token。它直接对应 storyline 中你想区分的那条线：prior work 关心的是 token-level compute allocation，而不是 trajectory-level budget belief。

7. [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (2024): 论文研究 LLM inference-time compute scaling，指出不同 prompt 难度适合不同 test-time strategy，并提出 compute-optimal 的按题分配方法。它是你在 related work 中区分 `performance given budget` 与 `knowing budget need` 时最重要的近邻之一。

8. [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) (2025): 提出非常直接的 test-time scaling 方法，包括小规模 reasoning trace 数据和 `budget forcing`，可以强制延长或截断“thinking”过程。它说明 LLM 社区已经开始把 token budget 当作可控推理资源，但仍是为了提升答案质量，而非输出 calibrated remaining-budget interval。

9. [Adaptive Stopping for Multi-Turn LLM Reasoning](https://arxiv.org/abs/2604.01413) (2026): 这篇工作直接提出 multi-turn reasoning / ReAct 式流程中的核心问题：“模型什么时候该停？”作者用 conformal prediction 构造 MiCP，在多轮过程中分配 error budget，使模型能在保持整体 coverage 的同时提前停止，减少 turn 数、推理成本与 prediction set 大小。它和你的工作很近，因为同样关注在线 stopping 与 trajectory 过程；但它保证的是答案覆盖率，而不是剩余 budget 的区间估计。

10. [Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use](https://arxiv.org/abs/2602.11541) (2026): 研究带严格 monetary budget 的 tool-augmented LLM agents，把问题形式化为带价格和随机工具执行结果的序列决策问题，并提出 INTENT，通过 intention-aware hierarchical world model 在线预判未来 tool 使用与 risk-calibrated cost。它是目前最接近你 financial-budget 设定的工作之一，但重点仍是在线规划与 hard budget feasibility，而不是估计 remaining budget interval。

11. [Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents](https://arxiv.org/abs/2602.16699) (2026): 论文关注 agent 在信息获取场景下如何权衡“继续探索的成本”和“过早提交答案的风险”，并通过 CTA 让模型显式推理这种 cost-uncertainty tradeoff，从而学会更优地测试、检索和停止。它和你的工作非常接近，因为都把 uncertainty 与 future cost 放进 agent 行为决策里；但它主要优化 explore/commit policy，不输出 calibrated cost-to-go interval。

12. [Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search](https://arxiv.org/abs/2603.08877) (2026): 这篇论文构建 BCAS 测试框架，在固定 tool-call 和 completion-token 预算下系统测量搜索深度、检索策略和 completion budget 对准确率与成本的影响。它的重要相似点在于显式 `surface remaining budget` 并用其约束 agentic search；但其核心是 measurement study 和 system configuration，而不是 agent 的内生预算估计能力。

## 4. LLM Self-Monitoring / Uncertainty / Reflection / Failure Awareness

1. [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) (2022): 研究模型能否评估自己答案是否为真，以及是否“知道”某个问题的答案；作者发现较大模型在合适格式下能给出相对可校准的 `P(True)` 与 `P(IK)`。这篇文献是你论证“foundation model 确实可能具备某种自我评估能力”的关键来源。

2. [Large Language Models Must Be Taught to Know What They Don't Know](https://arxiv.org/abs/2406.08391) (2024): 论文认为单靠 prompting 不足以得到可靠 uncertainty，需要用少量带正确/错误标签的数据进行轻量微调，才能获得泛化更好的不确定性估计器。它和你的 controlled intervention 逻辑很接近：更好的 calibration 可以改变 downstream use。

3. [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063) (2023): 系统比较 verbalized confidence、采样一致性与聚合等黑盒 uncertainty elicitation 方法，并在 calibration 与 failure prediction 上做基准评测。结论之一是 LLM 往往会 verbal overconfidence，这和你观测到的“整体偏乐观”高度一致。

4. [Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?](https://arxiv.org/abs/2405.16908) (2024): 提出“faithful response uncertainty”指标，衡量模型真实内在不确定性与语言表述中的犹豫程度是否匹配。结果显示现有对齐后的 LLM 仍不擅长忠实表达自己的 uncertainty，这正好支撑你对 verbalized confidence baseline 的动机。

5. [Towards A Unified View of Answer Calibration for Multi-Step Reasoning](https://arxiv.org/abs/2311.09101) (2023): 把 multi-step reasoning 分成 path generation 与 answer calibration 两阶段，并系统梳理 step-level 与 path-level calibration 方法。它能帮助你把“estimation 作为一个与最终决策可解耦的中间原语”讲得更清楚。

6. [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) (2023): 提出无需外部数据库的黑盒 hallucination 检测方法，利用同一 prompt 下多次采样的一致性来判断 factuality。其核心思想是“如果模型真的知道，样本间应更一致”，这和你文中 success/failure 轨迹上的置信差异有直接呼应。

7. [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) (2023): 用同一个 LLM 完成初始输出、反馈生成与迭代修正，不需要额外训练即可在多任务上提升结果质量。它展示了 self-feedback 能成为 agent 过程控制的一部分，但没有显式建模 budget cost。

8. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (2023): 通过自然语言形式的反思文本和 episodic memory，让 agent 根据 trial-and-error 反馈改进行为，而不是更新权重。它是“agent 有可能沿 trajectory 学会识别自己是否偏离目标”的代表工作。

9. [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738) (2023): 让模型调用外部工具对初始输出进行检查、再根据工具反馈修正结果，在 QA、代码和安全性任务上都能带来提升。它说明更强的外部校验能改善行为，但你的工作关心的是无需 oracle 工具时 agent 自己的 remaining-cost belief。

10. [Large Language Models have Intrinsic Self-Correction Ability](https://arxiv.org/abs/2406.15673) (2024): 论文从理论和实验两方面重新审视 intrinsic self-correction，指出 zero temperature 与 fair prompt 对发挥这类能力很关键。它有助于你论证：agent 的预算估计未必完全不可能，问题更像是触发条件和 calibration 机制不足。

11. [Devil's Advocate: Anticipatory Reflection for LLM Agents](https://arxiv.org/abs/2405.16334) (2024): 提出 anticipatory reflection，在行动前预想潜在失败、行动后检查偏差、完成后总结策略，从而提升 WebArena 表现并减少反复试错。它和你文中“fail 判断往往不是一开始就准确，而是后期才改善”这条故事线非常贴近。

12. [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271) (2024): 用 uncertainty-aware simulation、information-gain reward 和 reward propagation 让 LLM 更会在不确定场景下主动提问获取信息。它非常适合支撑你的元认知视角：uncertainty 不是副产品，而是影响行动选择的核心信号。

13. [Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents](https://arxiv.org/abs/2402.11651) (2024): 这篇工作指出 failed trajectories 不是噪声，而是 agent 调优的重要学习信号；简单地把轨迹标记为成功/失败就能提升性能。它和你关于 success/failure 非对称、失败轨迹尤其关键的论述高度一致。

14. [AgentRx: Diagnosing AI Agent Failures from Execution Trajectories](https://arxiv.org/abs/2602.02475) (2026): 论文发布了带关键失败步标注的 failed-agent trajectory benchmark，并提出自动诊断框架来定位 critical failure step 和 failure category。它和你的工作一样把长轨迹失败分析当成一等问题，但它输出的是失败归因与定位，而不是在线预算信念或剩余成本区间。

## 5. Agent Trajectory / Benchmark / Environment

1. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022): ReAct 把 reasoning trace 与 action sequence 交错生成，让模型在轨迹中持续更新计划并调用外部信息。它是你整篇论文里“trajectory-level online problem”最自然的 agent 背景文献之一。

2. [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206) (2022): 提供大规模、接近真实电商网页的交互环境，要求 agent 根据自然语言需求跨页面导航、检索和购买商品。它很好地说明多步 web trajectory 中 token、时间和行动成本都会动态累积。

3. [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854) (2023): 构造了更真实、可复现的网页任务环境，覆盖电商、论坛、协作开发和 CMS，任务长程且复杂。它能支撑你强调“真实部署不是单步问答，而是长轨迹在线决策”。

4. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) (2023): 给出 8 个交互环境组成的 agent benchmark，指出主流 LLM 的主要瓶颈在长程推理、决策与指令遵循。它与 BudgetBench 的区别恰好在于：前者主要看任务成功，后者进一步看 agent 是否知道自己还要花多少预算。

5. [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768) (2020): 通过把文本世界与具身视觉环境对齐，让 agent 先在抽象文本层学策略，再执行到具体场景。它提供了一个很好的例子：长轨迹 agent 的“计划、执行、更新”是连续过程，因此预算估计天然应是在线问题而非事后统计。

6. [Self-Monitoring Navigation Agent via Auxiliary Progress Estimation](https://arxiv.org/abs/1901.03035) (2019): 在视觉语言导航任务中引入 progress monitor，让 agent 在执行过程中估计自己沿 instruction 的完成进度，并把进度估计作为动作选择的辅助信号。虽然这不是 LLM budget paper，但它是“trajectory-level online self-monitoring”非常早且很贴切的先例。

7. [The Regretful Agent: Heuristic-Aided Navigation through Progress Estimation](https://arxiv.org/abs/1903.01602) (2019): 这篇工作进一步把 progress estimation 当作 search heuristic，用来决定 agent 是否前进、回退以及如何利用历史方向信息。它和你的论文很像的地方在于：估计信号不是最终任务目标本身，而是服务于轨迹控制和中途纠偏的中间认知变量。

## 可直接用于 related work 的总括句

可以从这批文献里提炼出一条很清楚的 related-work 论述：

1. `CMDP / safe RL / budgeted RL` 关心的是 **在已知或可学习的外部 cost / constraint 下做优化**。
2. `prediction interval / calibration` 关心的是 **给固定预测器输出可信区间，并权衡 coverage 与 sharpness**。
3. `test-time compute scaling` 关心的是 **给定预算时如何更聪明地花预算**。
4. `LLM self-evaluation / reflection` 说明 **模型并非完全没有元认知能力，但其 confidence 往往过度乐观、表达也不够 faithful**。
5. `agent benchmarks` 大多只看 **任务是否完成**，很少追问 **agent 在 trajectory 中是否知道自己还需要多少 budget**。

因此，你的工作最自然的定位不是这些方向的简单拼接，而是把它们之间尚未被系统研究的空白明确提出：  
**foundation model agent 的 trajectory-level online remaining-budget interval estimation**。
