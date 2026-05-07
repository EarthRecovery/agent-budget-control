# SearchR1 GPT-5.2 Instant Prompt Comparison

比较对象：

- Baseline: [searchr1-gpt5.2instant](./searchr1-gpt5.2instant/summary.md)
- New Prompt: [searchr1-gpt5.2instant-new-prompt](./searchr1-gpt5.2instant-new-prompt/summary.md)

两组结果使用相同 rollout 源：

- Rollout source: `${HOME}/database/origin/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json`

区别只在 estimation 输入：

- Baseline estimation: `${HOME}/database/estimation/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main.json`
- New prompt estimation: `${HOME}/database/estimation-test/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main2/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main2.json`

## Headline

新 prompt 的主要收益不在 `can_finish` 二分类，而在区间质量。

- `can_finish_accuracy` 不变：`0.6204 -> 0.6204`
- `first-turn accuracy` 不变：`0.672 -> 0.672`
- `remaining_token_interval_coverage_rate` 明显上升：`0.1366 -> 0.1975`
- `overall reward mean` 上升：`0.355 -> 0.377`
- success rollout 中的 hit 数显著上升：`28 -> 43`
- success rollout 中 overly conservative 明显下降：`64 -> 49`
- API 总 token 消耗下降：`353,652 -> 303,262`，减少 `50,390`，约 `14.2%`

## Metrics

| Metric | Baseline | New Prompt | Delta |
| --- | ---: | ---: | ---: |
| Rollouts | 128 | 128 | 0 |
| Successful rollouts | 87 | 87 | 0 |
| Failed rollouts | 41 | 41 | 0 |
| Estimation samples | 324 | 324 | 0 |
| API success rate | 1.0000 | 1.0000 | 0.0000 |
| Can-finish accuracy | 0.6204 | 0.6204 | 0.0000 |
| First-turn accuracy | 0.6720 | 0.6720 | 0.0000 |
| Interval coverage rate | 0.1366 | 0.1975 | +0.0609 |
| Overall reward mean | 0.3550 | 0.3770 | +0.0220 |
| Success-rollout interval predictions | 200 | 201 | +1 |
| Hit count in success rollouts | 28 | 43 | +15 |
| Overly optimistic count | 108 | 109 | +1 |
| Overly conservative count | 64 | 49 | -15 |
| Rollout cached-input share | 0.220 | 0.220 | 0.000 |
| Estimation cached-input share | 0.019 | 0.020 | +0.001 |
| API total tokens sum | 353652 | 303262 | -50390 |

## Interpretation

- 新 prompt 没有改变模型对“能否完成预算”的判别能力，所以 Figure 1 和 Figure 2 的主结论应基本不变。
- 新 prompt 明显改善了区间命中质量。最直接的证据是 `interval coverage rate` 提升了约 `6.1` 个百分点，同时 success rollout 中 hit 数从 `28` 提升到 `43`。
- 这次提升主要来自“少保守”。`overly conservative` 从 `64` 降到 `49`，而 `overly optimistic` 基本持平，说明新区间不是靠整体放宽来换命中。
- reward 提升与上面一致。因为 reward 同时奖励命中和区间紧度，所以 `0.355 -> 0.377` 说明新 prompt 在保持判别能力不变时，提高了区间有效性。
- 新 prompt 还更省 token。总 estimation token 从 `353,652` 降到 `303,262`，说明新格式不只是更准，而且更便宜。

## Figure Links

Baseline:

- [Figure 1](./searchr1-gpt5.2instant/figure1_first_turn_accuracy_confusion.png)
- [Figure 2](./searchr1-gpt5.2instant/figure2_all_turn_accuracy_by_relative_position.png)
- [Figure 3](./searchr1-gpt5.2instant/figure3_reward_curve_by_relative_position.png)
- [Figure 4](./searchr1-gpt5.2instant/figure4_hit_optimism_pessimism_in_success_rollouts.png)
- [Figure 5](./searchr1-gpt5.2instant/figure5_range_width_change_in_success_rollouts.png)
- [Figure 6](./searchr1-gpt5.2instant/figure6_cached_tokens_rollout_vs_estimation.png)
- [Figure 7](./searchr1-gpt5.2instant/figure7_average_tokens_used_in_rollout_turns.png)

New Prompt:

- [Figure 1](./searchr1-gpt5.2instant-new-prompt/figure1_first_turn_accuracy_confusion.png)
- [Figure 2](./searchr1-gpt5.2instant-new-prompt/figure2_all_turn_accuracy_by_relative_position.png)
- [Figure 3](./searchr1-gpt5.2instant-new-prompt/figure3_reward_curve_by_relative_position.png)
- [Figure 4](./searchr1-gpt5.2instant-new-prompt/figure4_hit_optimism_pessimism_in_success_rollouts.png)
- [Figure 5](./searchr1-gpt5.2instant-new-prompt/figure5_range_width_change_in_success_rollouts.png)
- [Figure 6](./searchr1-gpt5.2instant-new-prompt/figure6_cached_tokens_rollout_vs_estimation.png)
- [Figure 7](./searchr1-gpt5.2instant-new-prompt/figure7_average_tokens_used_in_rollout_turns.png)

## Bottom Line

对 SearchR1 GPT-5.2 Instant，这次 prompt 改动的效果可以概括为：

- 不改变 `can finish / cannot finish` 的判别准确率
- 明显改善 remaining-token interval 的命中质量
- 明显减少过度保守
- 在更低 estimation token 成本下拿到更高 reward
