Section 6. Related Work

1. Cost-constrained decision making and budgeted agents

Prior work on constrained and budgeted sequential decision making has mostly treated cost as an exogenous object: safe RL, CMDP, and budgeted-RL methods optimize reward subject to safety or resource constraints, with the cost function specified by the environment, estimated as part of the environment model, or certified during exploration [1,2,3,4,5,6,7,8]. A more recent line brings budget explicitly into agentic LLM systems by introducing token, tool-use, or monetary limits, budget-aware planning, and cost-aware exploration [9,10,11,12]. These works move closer to deployment concerns, but they still study how to act under a fixed or externally imposed budget, rather than whether the agent can estimate the budget it will still need from its current state. Our setting is related but distinct: instead of optimizing under an externally specified budget model, we ask whether the agent can estimate its own remaining budget need online, where future cost is endogenous because it depends on the agent's subsequent reasoning, tool calls, retries, and recovery behavior.

2. Prediction intervals, calibration, and uncertainty-aware estimation

Prediction-interval and calibration work provides the closest methodological precedent because it studies finite-sample coverage, interval sharpness, conformal validity, and post-hoc calibration for uncertainty-aware prediction [13,14,15,16,17,18,19]. These methods, however, are developed for fixed predictors on static supervised data, where the target distribution is not changed by the predictor's own future behavior [13,16,19]. In our setting, by contrast, the remaining-cost distribution evolves along the trajectory and is partly induced by the agent's own future actions. Our task is therefore closer to online belief tracking than to standard offline calibration, even though we inherit the language of coverage and tightness from this literature.

3. Adaptive compute, stopping, and test-time scaling

Another nearby line studies how much computation should be allocated to each input. Classic adaptive-compute, early-exit, and dynamic-routing methods decide when to stop or skip computation for efficiency [20,21,22,23,24,25], while more recent LLM work studies test-time compute scaling and multi-turn stopping under token budgets [9,26,27]. The common goal in this line is to improve performance given a budget, reduce wasted computation, or terminate reasoning when additional compute is no longer useful. Our question is different: not how to spend a fixed compute budget optimally, but whether the agent knows how much additional budget it will need from its current trajectory state, and how uncertain that estimate is.

4. Self-monitoring, failure awareness, and trajectory-centric agents

Work on LLM self-evaluation and uncertainty elicitation studies whether models can estimate answer correctness, verbalize confidence, or recognize when they do not know [28,29,30,31,32,33]. A related line uses reflection, self-feedback, and tool-mediated critique to improve downstream behavior once such signals are surfaced [34,35,36,37,38,39]. Recent work on negative trajectories and failure diagnosis further shifts attention from final task success to where and why agents fail along a trajectory [40,41]. At the environment level, interactive agent benchmarks and progress-monitoring methods establish the long-horizon settings where such intermediate signals matter [42,43,44,45,46,47,48]. Yet these works typically target correctness, repair, diagnosis, or progress, rather than whether an agent can forecast its remaining token or financial budget. Our work instead treats budget awareness as a first-class deployability capability and shows that the dominant failure mode is selective over-optimism on failed trajectories, precisely where early budget warning is most needed.

Figure 6. Delta map between our formulation and the closest prior work. Rows are representative neighboring papers; columns are the main components of the problem definition. `✓` means the component is directly modeled, `△` means partially related, and `✗` means largely absent.

| Similar Work | Agentic setting | Trajectory-level | Online during execution | Interval / calibrated UQ | Token budget | Financial budget | Endogenous future cost | Failure-specific analysis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Can LLMs Express Their Uncertainty? [29] | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | △ |
| Conformalized Quantile Regression [16] | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Adaptive Stopping for Multi-Turn LLM Reasoning [9] | △ | ✓ | ✓ | ✓ | △ | ✗ | △ | ✗ |
| Budget-Constrained Agentic LLMs (INTENT) [10] | ✓ | ✓ | ✓ | ✗ | △ | ✓ | ✓ | ✗ |
| Calibrate-Then-Act [11] | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | △ | ✗ |
| Budget-Constrained Agentic Search (BCAS) [12] | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | △ | ✗ |
| Our work | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The central gap highlighted by Figure 6 is that prior work typically covers only one or two sides of the space at a time: constrained-agent papers model budget-aware acting but not calibrated remaining-cost intervals; interval-prediction papers model coverage and sharpness but not agentic, trajectory-dependent, endogenous costs; and LLM self-monitoring papers study uncertainty or failure awareness but not token-plus-financial remaining-budget estimation. Our work is positioned at the intersection of these dimensions and formulates trajectory-level online remaining-budget interval estimation as a distinct capability of deployable foundation-model agents.

Appendix A. Extended Related Work

1. Cost-constrained decision making and endogenous cost

Prior work on constrained and budgeted sequential decision making studies how to optimize behavior under externally specified costs or safety constraints, typically assuming that the cost function is given by the environment, can be learned as part of the environment model, or can be certified during exploration, rather than asking whether the agent itself can estimate its remaining resource need online [1,2,3,4,5,6,7,8]. More recent LLM-agent work moves closer to our setting by introducing explicit token, tool-use, or monetary constraints, online stopping rules, and cost-aware exploration, but its primary goal is still to improve acting under a fixed budget or to enforce budget feasibility, not to evaluate whether an agent can produce a calibrated belief over the budget it will still need to finish the task [9,10,11,12]. In contrast, our work treats budget estimation itself as the task: the agent must continuously predict a remaining-budget interval along the trajectory, where future cost is endogenous because it is partly induced by the agent's own reasoning, tool calls, retries, and recovery behavior.

2. Prediction intervals and calibration for uncertainty-aware estimation

Prediction-interval and calibration work provides the closest methodological precedent for our formulation because it explicitly studies coverage, sharpness, conformal validity, and post-hoc calibration, but these methods are developed for fixed predictors on static supervised data rather than for agents whose future actions reshape the downstream cost distribution [13,14,15,16,17,18,19]. This distinction matters because in our setting the target is not a stationary regression label: the remaining cost distribution changes with the agent's own behavior, so the challenge is not only to calibrate an uncertainty estimate, but to do so online as the trajectory unfolds. Our evaluation therefore inherits the language of coverage and tightness from this literature, while extending it to long-horizon interactive rollouts and separating successful from failed trajectories.

3. Adaptive compute, stopping, and token-budgeted reasoning

A related line on adaptive compute and stopping decides how much inference to allocate to an input or when to terminate reasoning, early-exit, or layer execution, which is highly relevant to our token-budget motivation but still optimizes performance given budget rather than estimating budget need itself [9,20,21,22,23,24,25,26,27]. These methods show that compute should be distributed non-uniformly across inputs, and some of the recent LLM work even makes token budget or stopping policy explicit, but the output remains a control decision rather than a calibrated interval over future cost. Our work differs from this line by asking whether an agent knows how much more budget it will need, not merely whether it can spend a fixed budget more effectively.

4. Self-evaluation and uncertainty elicitation in LLMs

Work on self-evaluation and uncertainty elicitation shows that language models can sometimes estimate whether an answer is likely to be correct, whether they know the answer, or whether uncertainty should be verbalized, while reflection, self-correction, and tool-mediated critique can improve downstream behavior once such signals are surfaced [28,29,30,31,32,33,34,35,36,37,38,39]. These papers are highly relevant to our motivation because they suggest that foundation-model agents do possess partial metacognitive signals, but the target variable is usually answer correctness, information gain, or action repair rather than remaining cost. Our formulation reuses the same high-level intuition, but shifts the object of uncertainty from correctness to future resource need, and from single-shot answers to trajectory-level execution.

5. Failure diagnosis, negative trajectories, and selective overconfidence

Recent work on failed trajectories shifts attention from final success labels to trajectory-level failure diagnosis, root-cause attribution, or learning from negative rollouts [40,41]. This line is closely related to our motivation because it recognizes that failure is a trajectory-level phenomenon rather than a final binary outcome. However, it remains largely retrospective: it asks where the agent failed or how failed traces can improve training, rather than whether the agent could have known earlier that the remaining budget was insufficient. Our central empirical claim is therefore different: current agents do not fail uniformly at budget estimation, but selectively fail on the very trajectories where early warning is most valuable, producing overly optimistic intervals on failed rollouts until late in execution.

6. Agent benchmarks and trajectory-level self-monitoring

Interactive agent benchmarks establish the long-horizon, tool-mediated settings in which our problem becomes meaningful, since they require reasoning, acting, replanning, and memory over trajectories whose cost accumulates across turns rather than in a single forward pass [42,43,44,45,46]. Closest in spirit are trajectory-level progress-monitoring methods, which use an auxiliary estimate of task progress to guide search, backtracking, and action choice, thereby showing that intermediate self-monitoring variables can materially improve multi-step control even when they are not the final task objective [47,48]. However, prior benchmarks and progress-estimation methods primarily measure task completion, path efficiency, or progress toward the goal, whereas we evaluate budget awareness as a first-class deployability capability: an agent must know not only what to do next, but also how much token or financial budget it is still likely to consume, how uncertain that estimate is, and whether this belief becomes better calibrated as the trajectory unfolds.

Reference Index

[1] Constrained Policy Optimization  
[2] Reward Constrained Policy Optimization  
[3] Budgeted Reinforcement Learning in Continuous State Space  
[4] Safe Reinforcement Learning in Constrained Markov Decision Processes  
[5] Constrained Upper Confidence Reinforcement Learning  
[6] Constrained Variational Policy Optimization for Safe Reinforcement Learning  
[7] Near-optimal Conservative Exploration in Reinforcement Learning under Episode-wise Constraints  
[8] Safe Reinforcement Learning for Constrained Markov Decision Processes with Stochastic Stopping Time  
[9] Adaptive Stopping for Multi-Turn LLM Reasoning  
[10] Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use  
[11] Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents  
[12] Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search  
[13] Distribution-Free Predictive Inference For Regression  
[14] High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach  
[15] Accurate Uncertainties for Deep Learning Using Calibrated Regression  
[16] Conformalized Quantile Regression  
[17] Deep Evidential Regression  
[18] Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification  
[19] Evaluating and Calibrating Uncertainty Prediction in Regression Tasks  
[20] Adaptive Computation Time for Recurrent Neural Networks  
[21] BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks  
[22] SkipNet: Learning Dynamic Routing in Convolutional Networks  
[23] Shallow-Deep Networks: Understanding and Mitigating Network Overthinking  
[24] Learning to Stop While Learning to Predict  
[25] Learning to Skip for Language Modeling  
[26] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters  
[27] s1: Simple test-time scaling  
[28] Language Models (Mostly) Know What They Know  
[29] Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs  
[30] Large Language Models Must Be Taught to Know What They Don't Know  
[31] Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?  
[32] Towards A Unified View of Answer Calibration for Multi-Step Reasoning  
[33] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models  
[34] Self-Refine: Iterative Refinement with Self-Feedback  
[35] Reflexion: Language Agents with Verbal Reinforcement Learning  
[36] CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing  
[37] Large Language Models have Intrinsic Self-Correction Ability  
[38] Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models  
[39] Devil's Advocate: Anticipatory Reflection for LLM Agents  
[40] Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents  
[41] AgentRx: Diagnosing AI Agent Failures from Execution Trajectories  
[42] ReAct: Synergizing Reasoning and Acting in Language Models  
[43] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents  
[44] WebArena: A Realistic Web Environment for Building Autonomous Agents  
[45] AgentBench: Evaluating LLMs as Agents  
[46] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning  
[47] Self-Monitoring Navigation Agent via Auxiliary Progress Estimation  
[48] The Regretful Agent: Heuristic-Aided Navigation through Progress Estimation
