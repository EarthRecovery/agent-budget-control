## 5. Analysis

## 5.1. Estimation Ability Is Separable from Rollout Ability

We first examine whether strong task performance translates into strong budget estimation, and whether first-turn estimates are sufficient to characterize this ability. The current results suggest that the answer to both questions is no. Rollout ability and estimation ability are only partially coupled: a model can be a strong actor but a weak estimator, and a model that appears strong on the first estimation turn can later become weak when evaluated over the full trajectory. This means that first-turn estimation is at best an early hint rather than a reliable summary of online budget awareness.

The first piece of evidence is the instability between first-turn and all-turn feasibility judgment. Across benchmarks, the ranking of estimators changes once we move from a task-start snapshot to the full trajectory. For example, on `SWE-bench`, `Qwen` is the strongest first-turn classifier, `Gemini` becomes the strongest all-turn classifier, while `GPT-5.2` is the strongest interval estimator. This ranking reversal is exactly what we would expect if budget estimation were an online process that improves, degrades, or changes character as more state information becomes available.

The second piece of evidence is the decoupling between task success and estimation quality. On `SearchR1`, `Opus` is the strongest actor (`75.8%` rollout success), yet `Sonnet` is the strongest interval estimator (`36.5%` success-case interval hit versus `23.1%` for `Opus`). On `Warehouse`, this separation is even sharper: all five models achieve `100%` rollout success, but joint interval hit still spans from `28.8%` down to `7.5%`. In other words, solving the task and knowing how much budget remains are empirically different abilities.

This suggests a more precise interpretation of online estimation. First-turn estimation mostly reflects prior beliefs about task difficulty before substantial evidence accumulates. All-turn estimation, by contrast, reflects whether the model can update that belief as the trajectory unfolds. A good analysis section should therefore treat first-turn metrics as a prior-quality diagnostic and all-turn metrics as an online belief-update diagnostic, rather than collapsing them into a single feasibility score.

Takeaway. Budget estimation should be understood as an online belief-updating capability rather than as a byproduct of actor strength.

![RQ1 Figure 1](../../../figure/agent-budget-control/overall/finding2_first_turn_vs_all_turn.png)

*Figure 5.1. First-turn and all-turn feasibility rankings are not stable. The estimator that looks strongest at task start is often not the strongest once the full trajectory is taken into account.*

![RQ1 Figure 2](../../../figure/agent-budget-control/overall/finding9_actor_vs_estimator.png)

*Figure 5.2. Rollout success and estimator strength are visibly decoupled. Strong actors are not consistently strong estimators, and the separation holds across all benchmark families.*

## 5.2. Bias Direction Is Structured Rather Than Random

We next ask why some models systematically underestimate remaining budget while others overestimate it, and whether these errors are random or structurally organized. The results favor the second interpretation. Bias direction is primarily benchmark-dependent, with model capability modulating how that benchmark-level pressure is expressed. `SearchR1` and `SWE-bench` are mainly optimistic; `Sokoban` exhibits a capability-dependent split between conservative stronger estimators and optimistic weaker ones; and `Warehouse` adds a third failure mode in which models predict `impossible` too early even when the rollout is still finishable.

At the benchmark level, the cleanest pattern is that bias direction clusters by environment. This is visible in the aggregate bias-driver figure: benchmarks do not merely differ in error magnitude, but in the dominant *type* of miss. `SearchR1` and `SWE-bench` are dominated by optimistic misses, meaning the upper bound of the predicted interval often still falls below the actual remaining budget. This suggests that models in these environments systematically fail to represent unresolved future work.

`Sokoban` shows a more subtle phenomenon. Here, bias direction depends strongly on estimator strength. The stronger interval estimators such as `Opus` and `Sonnet` skew conservative, while weaker estimators such as `Gemini` and `Qwen` flip optimistic. A plausible interpretation is that stronger models better recognize irreversible risk and therefore hedge upward, whereas weaker models misread local progress as evidence that the puzzle is nearly solved. Under this reading, optimism is not a sign of confidence but of incomplete difficulty modeling.

`Warehouse` introduces a qualitatively different asymmetry. Instead of simply shifting intervals upward or downward, some models fail at the gating stage itself by predicting `impossible` too early. `GPT-5.2` labels early actually-finishable states as impossible `72.7%` of the time, which indicates that part of the bias problem lies upstream of interval calibration. In other words, the model can fail before it ever reaches the question of "how much budget remains."

Taken together, these results suggest that optimism and conservatism are not best understood as generic calibration noise. They are structured error regimes. Some come from underestimating unresolved future work, some from over-hedging against irreversible dynamics, and some from prematurely collapsing feasibility altogether.

Takeaway. Budget estimation errors form task-conditioned bias regimes rather than random calibration noise.

![RQ2 Figure 1](../../../figure/agent-budget-control/overall/finding13_bias_direction_drivers.png)

*Figure 5.3. Bias direction is primarily a benchmark effect. Different tasks induce different dominant miss directions, and stage effects only modulate this higher-level structure.*

![RQ2 Figure 2](../../../figure/agent-budget-control/overall/finding6_sokoban_conservative_bias.png)

*Figure 5.4. Sokoban reveals a capability-dependent split: stronger estimators skew conservative, while weaker ones become optimistic. This indicates that bias direction itself carries information about estimator competence.*

## 5.3. Strong Estimators Are Defined by Calibration, Not Feasibility Judgment

If rollout success and feasibility classification are not sufficient to explain estimation quality, what then distinguishes a strong estimator from a weak one? The present results suggest that strong estimators should be understood as well-calibrated interval predictors rather than simply strong binary judges. Across the 15 token-budget model-task points, success-case interval hit correlates only weakly with feasibility macro-F1 (`r ≈ 0.36` first-turn; `r ≈ 0.35` all-turn), but more strongly with midpoint bias (`r ≈ -0.67`) and width adequacy (`r ≈ 0.62`). What separates strong estimators is therefore not merely the ability to say `can_finish` or `impossible`, but the ability to place the interval center near the truth while keeping the interval wide enough to cover real variation without collapsing.

The strength-signal figure makes this point directly. Binary feasibility metrics have only modest correlation with success-case interval hit, and rollout success is also an imperfect proxy. By contrast, calibration-oriented quantities such as midpoint bias and width adequacy line up much more closely with interval performance. This is the main reason why some models that appear strong under coarse task metrics still fail badly as estimators.

The phenotype map gives a more intuitive geometric view. Strong estimators cluster in the low-bias, non-collapsed-width region. Weak estimators occupy two failure corners: either they have badly shifted midpoints, or they produce intervals whose width has collapsed relative to true outcome spread. `Qwen` on `SearchR1` and `SWE-bench` is a particularly clear example of collapse: feasibility judgment may remain non-trivial, but interval prediction becomes nearly unusable because the estimates are both too low and too narrow.

This result has a methodological implication for the paper. If we want to evaluate or train budget-aware agents, the right target is not "higher feasibility F1" alone. The more informative target is a joint estimator phenotype: low midpoint bias, sufficient width, and strong online coverage. Put differently, a strong estimator is not merely a good classifier with numbers attached; it is a calibrated uncertainty model over remaining resource consumption.

Takeaway. Interval calibration, rather than binary feasibility judgment, is the defining property of estimator quality.

![RQ3 Figure 1](../../../figure/agent-budget-control/overall/finding14_estimator_strength_signals.png)

*Figure 5.5. Calibration signals explain interval quality better than binary or actor proxies. Feasibility metrics are only weakly correlated with success-case interval hit, while midpoint bias and width adequacy are substantially more informative.*

![RQ3 Figure 2](../../../figure/agent-budget-control/overall/finding15_token_phenotype_map.png)

*Figure 5.6. Strong token-budget estimators cluster in the low-bias, non-collapsed-width corner. This supports the view that estimator strength is a calibration phenotype rather than a binary-judgment phenotype.*

## 5.4. Warehouse Exposes a Coupled Multi-Budget Estimation Problem

The final question concerns the interaction among `time`, `cost`, and `warehouse_item_weeks` in the `Warehouse` setting. The results strongly support the view that this is not merely three independent one-dimensional estimation tasks. Instead, `Warehouse` poses a coupled feasibility-and-calibration problem with two distinct bottlenecks. First, the model must avoid declaring `impossible` too early. Second, conditional on believing the task is still finishable, it must jointly calibrate three constrained budgets whose error profiles are not symmetric. Single-dimension competence does not naturally transfer to joint coverage.

The dimension-specialization figure shows that the three budgets are not equally hard and are not mastered by the same model. `Opus` is strongest on `time` and `warehouse_item_weeks` (`41.8%` and `42.1%` strict hit), while `Sonnet` is strongest on `cost` (`33.1%`). This already suggests that the three dimensions expose partially distinct reasoning skills. Time appears closer to rollout progress, warehouse occupancy captures state accumulation, and cost behaves as the most globally compounding dimension.

The joint-gap figure then shows why single-dimension progress is not enough. Even when a model achieves non-trivial hit rates on individual dimensions, full all-dimension coverage drops sharply. `Opus` has the best joint hit at `28.8%`, yet this is still meaningfully lower than its average single-dimension hit. For weaker models, the gap is much worse. This "joint calibration tax" is the clearest sign that the three budgets interact rather than decompose cleanly.

This leads to a useful decomposition of the `Warehouse` difficulty. The first problem is *feasibility gating*: can the model recognize that the task is still finishable, instead of predicting `impossible` too early? The second problem is *multi-head coordination*: can the model produce intervals for `time`, `cost`, and `warehouse_item_weeks` that are individually plausible and jointly consistent with the same future rollout? The data suggests that many failures arise because models may succeed on one of these two layers while failing on the other.

For the paper narrative, `Warehouse` is therefore valuable not merely as an external-budget benchmark, but as evidence that budget awareness becomes qualitatively harder once resource dimensions are coupled. The multi-budget case does not simply add more numbers; it exposes coordination failures that are invisible in one-dimensional token estimation.

Takeaway. Multi-budget estimation introduces a joint calibration tax beyond single-dimension prediction.

![RQ4 Figure 1](../../../figure/agent-budget-control/overall/finding11_warehouse_dimension_specialization.png)

*Figure 5.7. The three Warehouse budgets show model specialization by dimension. Different models lead on `time`, `warehouse_item_weeks`, and `cost`, indicating that the three sub-problems are not identical.*

![RQ4 Figure 2](../../../figure/agent-budget-control/overall/finding12_warehouse_joint_gap.png)

*Figure 5.8. Joint multi-budget coverage is much harder than single-dimension coverage. This gap is the empirical signature of coupled-constraint estimation rather than independent scalar prediction.*
