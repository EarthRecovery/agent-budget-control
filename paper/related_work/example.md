Section 6. Related Work
1. Reasoning collapse and policy degeneracy in closed-loop LM / Agent RL training

Prior work on LLM-agent reinforcement learning has reported several collapse phenomena in closed-loop training [7, 61]. These include reasoning collapse, where rationales, plans, or explanations become increasingly templated and less tied to the input [61, 63, 69], and policy-level degeneracy, where the agent’s behavior concentrates around a small set of easy-to-reproduce action patterns [9, 59]. These phenomena are also related to model collapse in self-training, where models trained repeatedly on their own generated data can lose distributional diversity even when average metrics appear stable [12, 47].

2. Evaluating reasoning diversity, input dependence, and reasoning faithfulness

Existing diversity metrics often measure whether model outputs differ from one another, but they do not necessarily test whether those differences are systematically driven by the input [53, 69]. Common metrics include lexical-overlap measures such as n-gram statistics and self-BLEU [20, 74], embedding-based dispersion or distributional distance [33, 53], and uncertainty-based analyses [27, 43]. These metrics mainly capture within-input variability and may miss whether reasoning changes meaningfully across different inputs [43, 53]. Recent work has also studied input dependence through behavioral tests [11, 37, 73] and retrieval-style matching or prompt-reconstruction methods [28, 10, 71, 19]. Another related line of work studies reasoning faithfulness, asking whether explanations reflect the model’s true decision basis [18, 54, 48, 70]. In contrast, this paper focuses on whether RL-trained reasoning becomes less input-sensitive over time.

3. Stabilizing multi-turn Agent RL under closed-loop sampling

Prior work on stabilizing RL for language models and agents includes KL control, entropy regularization, clipping, reward shaping, curriculum learning, replay mixtures, rejection sampling, and best-of-N selection [40, 42, 41, 13, 49, 32, 36, 50, 9, 59, 57, 62, 63]. For multi-step agents, researchers have also used stepwise rewards, intermediate supervision, imitation-to-RL pipelines, and self-correction or reflection signals [4, 30, 55, 24, 46, 56, 65, 8, 61]. However, these methods do not necessarily prevent drift toward input-agnostic templates. If rollouts receive similar rewards regardless of reasoning quality, the gradient carries little information about which reasoning path matters [29, 31, 47, 69]. The paper therefore adopts an SNR-based perspective and uses reward variance to filter low-signal samples.

Appendix A. Extended Related Work
1. Reasoning collapse and policy degeneracy

The extended related work defines a broader family of degradation phenomena in closed-loop LLM-agent RL [7, 61]. After repeated updates on self-sampled trajectories, the model may gradually exhibit reasoning collapse and policy-level degeneracy [7, 61]. Reasoning collapse mainly refers to rationales, plans, or explanations becoming templated, less diverse, and less aligned with the input goal [61, 63, 69]. Policy-level degeneracy refers to behavioral choices becoming concentrated on a small set of repetitive action patterns, with reduced exploration and error correction [9, 59].

2. Connection to model collapse in self-training and synthetic-data training

This degradation family is connected to prior findings in self-training, self-distillation, and iterative fine-tuning on synthetic or model-generated data. When a model repeatedly trains on its own generated distribution, the feedback loop can narrow the effective data distribution, amplify high-probability modes, and suppress long-tail behaviors, even when average quality metrics remain stable [12, 47]. In agent RL, closed-loop optimization on on-policy trajectories introduces similar risks, but the degradation may first appear in language-level reasoning rather than final behavior [61, 62]. Reasoning-level degeneration may therefore decouple from, or even precede, policy-level degeneration [59].

3. Limitations of existing reasoning-diversity metrics

Prior work on reasoning diversity often asks how different outputs are, but not whether those differences are caused by different input goals [53, 69]. Existing metrics include n-gram statistics and self-BLEU [20, 74], embedding-based dispersion and distributional distance [33, 53], token-level uncertainty, and multi-sample coverage or consistency analysis [27, 43]. These metrics mainly capture randomness or within-input variability, but they are less sensitive to whether reasoning distributions shift coherently across inputs [43, 53]. They may also conflate true prompt-conditioned variation with prompt-agnostic surface diversity, especially when outputs converge to shared formats [17, 69].

4. Input dependence, prompt robustness, and retrieval-style matching

Recent work has begun to test whether model outputs preserve information about the input. Examples include behavioral tests and local decision-boundary checks [11, 37], prompt robustness benchmarks [73], and retrieval-style output-input matching or prompt reconstruction signals [28, 10, 71, 19]. However, the paper argues that there is still no unified and scalable framework tailored to closed-loop agent RL, especially under long-horizon training and collapse dynamics [9, 63].

5. Difference from reasoning faithfulness

A closely related line of work studies reasoning or explanation faithfulness, which asks whether a rationale reflects the true basis of a model’s decision rather than serving as a plausible post-hoc explanation [18, 54, 48, 70]. The paper’s question is related but distinct: it focuses on whether reasoning becomes less sensitive to the input during closed-loop RL and drifts toward reusable templates, even when local explanations remain self-consistent [17]. This motivates the paper’s decomposition of reasoning diversity into within-input variability and cross-input dependence.

6. Stabilizing multi-turn Agent RL and the SNR perspective

Prior stabilization methods for LLM and agent RL include KL control, trust-region constraints, entropy regularization, clipping and normalization, reward shaping, credit assignment, curriculum design, replay or offline-online mixtures, rejection sampling, and best-of-N selection [40, 42, 41, 13, 49, 32, 36, 50, 9, 59, 57, 62, 63]. For multi-step agents, prior work also explores stepwise rewards, intermediate supervision, imitation-to-RL pipelines, and self-correction or reflection signals [4, 30, 55, 24, 46, 56, 65, 8, 61]. The paper argues that these methods mainly target optimization stability or reward improvement, but they do not necessarily prevent prompt-agnostic template drift when the learning signal is weak or noisy [29, 31, 47, 69]. Its proposed SNR view uses within-prompt reward variance as a proxy for signal strength and filters low-signal samples to preserve input-conditioned reasoning [38, 44, 52, 9, 63].