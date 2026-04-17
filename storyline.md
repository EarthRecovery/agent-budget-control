Agent Budget Estimation - Storyline
基于当前已有图的判断
整体区分基本合理，但对当前这组图来说，层次略多。现有图最强支撑的是一条更集中的经验主线：这是一个 trajectory-level 的在线估计问题，并且当前模型呈现出三类非常清晰的现象：fail 判断会随轨迹推进而改善、success rollout 上区间会逐步变窄、整体误差类型偏过于乐观。相比之下，baseline taxonomy、oracle 上界、controlled intervention 这些段落目前没有对应图，保留为文字可以，但不适合作为主线的视觉支撑中心。

建议的主线拆分与配图
1. 这是一个在线而非事后统计的问题：budget 使用会沿 trajectory 动态变化，因此必须看过程而不是只看最终总成本。
配图：Figure 7（average token used in each rollout turn）。

2. 只看 first-turn estimation 不够，真正重要的是整条轨迹上 estimation 如何随相对位置变化。
配图：Figure 2（estimation accuracy in all turns）。
补充图：Figure 1 可作为 first-turn snapshot 放在旁边或 appendix。

3. 在 fail 相关判断上，模型并不是一开始就知道自己会失败，而是到后期判断准确率才明显升高。
配图：Figure 2（重点看 failure rollout 曲线）。

4. 在 success rollout 上，随着轨迹推进、信息增多，预测区间整体逐步收窄，说明模型有一定在线校准能力。
配图：Figure 5（range width change in success rollouts）。

5. 当前模型最突出的系统性偏差不是“瞎猜”，而是“过于乐观”：更常低估所需 budget，而不是高估。
配图：Figure 4（hit rate and miss direction in success rollouts）。

6. 以上几个现象共同体现在 reward 上：成功与失败轨迹的 reward 结构明显不同，而且 reward 的变化不是均匀的。
配图：Figure 3（reward curve）。

7. Figure 6 更适合放在 implementation / systems observation 中，而不是主发现里。它解释的是 rollout 和 estimation 的 cache 复用差异，对核心 scientific claim 是补充，不是主证据。

今天的 Agent 已经会规划、会调用工具、会执行复杂任务，但真实部署从来不是在无限 budget 下进行的。一个真正可部署的 Agent，不仅要会做事，还必须知道自己还要花多少资源，以及自己对这个判断有多确定。
这正是我们要研究的问题：Agent Budget Estimation。
为什么这个问题重要
Budget 不是部署时的附属约束，而是决定 Agent 能否落地的基本条件。许多 Agent 的失败并不是执行能力在某一刻突然崩溃，而是它在较长轨迹中持续错误估计剩余成本，未能及时意识到任务在当前预算下已不可完成。
但这个问题长期没有被系统研究。现有 benchmark 通常只看任务是否成功，最多事后统计成本，却很少问 Agent 在过程中是否知道自己还需要付出多少代价。这个区别是关键的：最终成本只描述了任务事后消耗了多少资源，却不能刻画 Agent 在执行过程中是否拥有足够的信息去判断"继续执行、调整策略，还是提前终止"。真正决定部署行为的，是对剩余成本的在线信念，而不是事后日志中的总成本。
为什么现有框架不够
已有的 constrained MDP 和 safe RL 框架虽然也处理成本约束，但它们通常假设 cost model 已知或可由环境直接提供。近年来 inference-time compute scaling 和 token-budget-aware reasoning 等工作虽然关注了推理开销的分配，但它们的目标是优化固定预算下的性能，而非评估 Agent 是否知道自己还需要多少预算。
Foundation Model Agent 在这里提出了一个结构性的新问题，而非旧框架的简单延伸。核心区别在于：Agent 自身既是执行者、也是成本的主要来源。 Token 生成、工具调用、重试策略的选择都动态改变着剩余成本分布，因此 estimation 必须是 Agent 的内生能力，而不是外部给定的约束参数。具体而言：
在传统 CMDP 中，cost model 是 exogenous 的——环境给定 cost function，agent 在其上做优化。而 Foundation Model Agent 的 cost 是 endogenous 的——agent 的 reasoning trace 本身就是成本的主要构成。
在传统 UQ 中，model 是 fixed 的——我们对一个不变的预测器做校准。而这里 agent 的行为本身在不断改变剩余 cost 的分布。
Inference-time scaling 优化的是 performance given budget。我们评估的是 agent 是否 knows its budget need。
这些特性使得"Foundation Model Agent 的在线预算估计"成为一个值得独立研究的问题。
Formulation
我们将 budget estimation 形式化为一个 trajectory-level online estimation 问题：Agent 需要在执行过程中持续判断完成剩余任务还需要多少预算——不是二元的可行/不可行，而是给出剩余预算的预测区间。
为什么是区间而非点估计
两个状态即使具有相同的期望剩余成本，其不确定性不同时，对"是否仍应继续执行"的最优决策可能完全不同。区间估计捕获的正是这个不确定性。
为什么不直接学一个 cost-to-go value function 或 stopping policy
Value function 或 stopping policy 直接输出决策（"继续"或"终止"），但它们把 risk tolerance 硬编码进了训练目标——换一个部署场景、换一个预算阈值，整个策略就需要重新训练。区间估计是一个更基础的认知原语（epistemic primitive）：它输出的是 Agent 对"还需要多少成本"的信念状态，部署方可以根据自身的 risk preference 在这个信念之上叠加任意决策规则。换言之，区间估计与下游决策解耦，同一个 estimation 能力可以跨部署场景复用。
Foundation Model Agent 带来的独特机会
值得强调的是，区间估计在传统 UQ 中并不新鲜，但 Foundation Model Agent 提供了独特的条件：（a）这些 Agent 具备自然语言推理能力，因此可以被要求对自身剩余成本进行 self-reflective estimation，这在传统 agent 中不可行；（b）budget 同时跨越 token 消耗和真实财务成本两个层面，使 estimation 不再是单一维度的问题；（c）Agent 的 reasoning trace 既是任务执行的载体、也是成本的来源，二者深度耦合。
BudgetBench
基于这一 formulation，我们构建 BudgetBench 作为研究在线预算区间估计的统一实验平台。其设计沿三个轴追求高覆盖度：
Budget 类型。 同时覆盖 token budget（模型"想了多少"）和 financial budget（action 在现实世界执行后产生的资源代价，基于真实仓库管理场景开发）。两类 budget 承担互补角色：token budget 测的是 Agent 能否预估自身推理消耗（内部成本意识），financial budget 测的是它能否预估行动在现实世界中造成的代价（外部成本意识），二者共同构成可部署 Agent 的完整 budget awareness。
任务领域。 包括 multi-step planning 与 environment interaction（Search / Code / Sokoban / Webshop / Robotoullie）以及 supply-chain resource allocation。
难度与成功率分布。 我们刻意不把所有环境人为卡到 50% success rate。真实部署中存在 5% 和 95% 成功率的任务，benchmark 应当反映这种分布而非抹平它。
Evaluation Framework
我们评估的不只是最终误差，而是区间估计在整条轨迹上的 coverage 与 tightness，以及这些性质如何随执行推进而动态演化。这使得我们能够区分不同的失效模式："估得不准""过度自信"与"过于保守"。Coverage 决定 Agent 是否系统性漏掉真实所需成本，tightness 决定它能否在保持风险可控的同时给出足够有用的决策信号。对部署而言，一个过于乐观但窄的区间，往往比一个宽但校准良好的区间更危险。
Baselines
由于 online budget interval estimation 是一个新定义的 task，不存在现成的 direct competitor。为系统地理解这个问题，我们构造四类 baseline，每一类回应一个不同的自然问题。
第一类：Naive Estimation Baselines（这个问题是不是平凡可解的？）
Fixed-ratio estimator： 假设剩余成本与已完成步骤数成固定比例。
History-mean estimator： 用同类任务的历史平均成本作为常数预测。
Linear extrapolation： 用当前已消耗的 cost 线性外推至任务完成。
如果 LLM-based estimator 连这些都赢不了，说明 self-reflection 并没有提供信息增益。
第二类：LLM Prompting Variants（不同 prompting 策略在 coverage/tightness 上有什么 trade-off？）
Zero-shot prompting： 直接问模型"你觉得还要花多少 token"。
Chain-of-thought estimation： 让模型先分析剩余步骤再给数字。
Verbalized confidence： 让模型同时输出 point estimate 和 self-reported confidence，然后将 confidence 转换成区间宽度。
第三类：Post-hoc Calibration Methods（为什么不直接用现有 UQ 方法？）
Conformal prediction（简化版）： 在 held-out trajectories 上收集 residuals，用 quantile 构造 prediction interval。
Temperature scaling / Platt scaling： 对模型的 point estimate 做 post-hoc 校准。
这组 baseline 的关键作用是说明：即使使用了标准 UQ 工具，offline calibration 也难以解决"失败轨迹上选择性过度自信"的问题——因为 failure trajectories 在校准集中本身就是 minority，post-hoc 方法会被 success trajectories 主导。
第四类：Oracle / Skyline（理论上界在哪里？）
Oracle estimator： 拥有 ground-truth remaining cost，始终输出真实剩余成本 ± 固定 margin。
这给出了 coverage/tightness 的上界，让读者知道当前方法离天花板还有多远。
核心发现：选择性失效
实验揭示了一个非直觉的不对称现象：当前模型并非在所有情况下都估不好。
在成功轨迹上， 模型的区间会随执行推进逐步收缩并最终覆盖真值，表现出合理的在线学习能力。
在失败轨迹上， 模型直到很晚仍给出乐观且漏掉真值的窄区间。
换言之，模型的 estimation 能力不是均匀地差，而是选择性地失效：恰恰在最需要发出预警的场景下，它反而最"自信"。这种不对称性意味着，单看平均校准指标会严重低估部署风险——真正的危险集中在模型未能识别出自己正在失败的那些轨迹上。
这一发现挑战了一个隐含假设：既然模型在成功轨迹上能逐步校准，那么 budget awareness 的缺失只是"能力不够"的问题，更多数据和更强模型自然会解决。但我们观察到，即使在成功率很高的环境中（>90%），模型对少数失败轨迹的 estimation 依然系统性地过度乐观。这说明问题不在于整体能力不足，而在于模型缺乏对"当前执行是否偏离正轨"的元认知信号。
当我们将评估从单一 token budget 扩展到 financial budget 时，这种选择性失效不仅没有消失，反而被放大——financial cost 的 estimation 更难，而模型在这一维度上的过度自信更为严重。这表明多维预算不是另一个独立问题，而是在暴露 agent 原本就存在的元认知盲区。
Controlled Intervention 实验
为提供 estimation quality 与 downstream behavior 之间关联性的受控证据，我们设计了 minimal intervention 实验：固定相同的 base agent，仅替换其 estimation module 的校准程度（通过简单的 SFT 训练三个梯度的 estimator），观察 budget compliance 和 strategy adaptation 的变化。
我们刻意不引入复杂的训练方法，因为本文的目标不是提出最优 solution，而是用最小干预表明 estimation 是一个可操作的节点——校准度每提升一个梯度，Agent 在预算不足时的提前终止/重规划率显著上升，budget violation 率相应下降。
这一点尤为重要：更好的 budget awareness 并非以牺牲任务性能为代价，而是让 Agent 在"做不了"的时候更早停下来、把资源留给"做得了"的任务。
总结
本文揭示的核心发现是：当前 Agent 的 budget estimation 存在选择性失效——它们并非全面地不知道代价，而是恰恰在最需要预警的失败轨迹上系统性地过度自信。我们提供：
BudgetBench 作为诊断平台；
Trajectory-level coverage/tightness 作为评估框架；
四类结构化 baselines 从不同角度建立参照系；
Controlled intervention 证据 表明校准的 estimation 可以切实改善部署行为。
这些工作共同将 budget estimation 从事后统计指标重新定位为 Agent 可部署性的核心能力维度。
