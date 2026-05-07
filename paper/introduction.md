\section{Introduction}

Foundation model agents are increasingly deployed in longer horizons and higher-stakes tasks: a coding agent consumes tokens per reasoning step, a web agent spends API calls per search query, and a supply-chain agent commits real dollars and warehouse capacity per procurement decision. In this process, both (1) the cost their generation consumes, and (2) the cost their actions commits, are growing rapidly. A natural solution is to make agents cheaper: compress traces, prune tool calls, distill into smaller models. But cost reduction optimizes a single objective of \textbf{spend less} and ignores a more fundamental question: \emph{does the agent know how much budget it needs?} An agent that cannot estimate its own resource requirements cannot decide when to abort a hopeless task, when to request more resources, or how to allocate budget across sub-goals.

Two gaps prevent systematic study of this capability. \textbf{First,} agent benchmarks usually calculate token consumptions objectively, but hardly ever ask whether the agent \emph{knew} what it would cost. \textbf{Second,} most evaluation protocol evaluates \textit{single point} prediction at \textit{task start}. This can be brittle and unrealistic for long-horizon agentic tasks, and a human analogy is like this: a project manager should re-estimate the remaining timeline at every milestone, gives a range rather than a point, and flags when completion becomes infeasible. 

To address these limitations, we introduce \textsc{BudgetBench} that asks agent to estimate their budget. 
Our benchmark focus on evaluation that is \textit{confidence-aware} and \textit{progressive throughout execution}.
Specifically, we record a full agent rollout without any budget constraint, then queries the same agent separately at every turn: \emph{given current progress, how much budget remains to finish? Provide an interval with confidence, or declare the task impossible.} 
\textsc{BudgetBench} evaluate two fundamentally different budget modalities: (1) \emph{Internal budget}: primarily token consumption, generated whenever the model runs; (2) \emph{External budget}: cost incurred by the agent's environment actions, such as money and time.
We evaluate \textit{internal budget} across tasks spanning information retrieval, planning, and software engineering, and evaluate \textit{external budget} in a Warehouse Management environment with three mutually-constrained budget dimensions (cost, time, and inventory), curated from real enterprise data.



Throughout this evaluation in five frontier models, we report four findings below:

\textbf{Budget estimation decomposes into three sub-capabilities, and models are inconsistent across them.} We measure binary feasibility (can the task be finished?), failure warning (can infeasibility be flagged early?), and interval calibration (is the predicted range accurate?). No model wins all three. On Sokoban, Gemini achieves the highest all-turn macro-F1 (61.9\%) but only 8.8\% interval hit rate. On Warehouse, Qwen leads both all-turn macro-F1 and failure warning, yet its joint interval hit is 7.5\%.

\textbf{Task performance does not predict estimation quality.} On SearchR1, Opus is the strongest actor (75.8\%) but Sonnet is the strongest interval estimator (36.5\% interval hit vs.\ Opus's 23.1\%). On Warehouse, all models achieve 100\% task success, yet joint interval hit ranges from 28.8\% down to 7.5\%. This separation appears across all four benchmarks.

\textbf{Failure modes are structured and vary with task and model capability.} On SearchR1, all five models exhibit optimistic bias; Qwen shows near-total collapse (81 optimistic misses, 0 hits). On Sokoban, a capability-dependent split emerges: stronger interval estimators skew conservative while weaker ones flip optimistic --- insufficient understanding of task complexity paradoxically produces underestimation. On Warehouse, a third pattern appears: GPT-5.2 predicts impossible on 72.7\% of early actually-finishable states.

\textbf{What separates good estimators is interval calibration, not binary judgment.} Across the 15 token-budget model--task points, success-case interval hit correlates only weakly with feasibility macro-F1 ($r \approx 0.36$ first-turn; $r \approx 0.35$ all-turn), but more strongly with midpoint bias ($r \approx -0.67$) and width adequacy ($r \approx 0.62$). Binary accuracy --- the most obvious metric --- has substantially less explanatory power than calibration quality.
