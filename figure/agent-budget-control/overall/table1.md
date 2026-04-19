# Table 1

Within each benchmark, the best value in each metric column is marked in red.
`Avg Turns` uses lower-is-better; other highlighted numeric columns use higher-is-better. `Estimations (S/F)` is highlighted by total estimation count.

| Benchmark | Model | Rollout Success | Avg Turns | First-Turn Success Pred Acc | Estimations (Succ/Fail Rollouts) | Pred Hit on Success Rollouts | Pred Hit on Fail Rollouts | Success-Rollout Interval Hit | Success-Rollout Reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SearchR1 | Claude Opus 4.7 Low Thinking | <span style="color:red">75.8%</span> | <span style="color:red">1.78</span> | 65.2% | 87 (52/35) | <span style="color:red">100.0%</span> | <span style="color:red">2.9%</span> | 23.1% | 0.114 |
| SearchR1 | Claude Sonnet 4.6 Low Thinking | 71.1% | 1.87 | 60.9% | 102 (52/50) | 98.1% | 0.0% | <span style="color:red">36.5%</span> | <span style="color:red">0.154</span> |
| SearchR1 | GPT-5.2 Instant | 68.0% | 3.69 | <span style="color:red">67.2%</span> | <span style="color:red">324 (201/123)</span> | <span style="color:red">100.0%</span> | 0.0% | 21.4% | 0.031 |
| Sokoban | Claude Opus 4.7 Low Thinking | <span style="color:red">56.2%</span> | <span style="color:red">5.04</span> | <span style="color:red">61.5%</span> | 355 (207/148) | <span style="color:red">100.0%</span> | 8.2% | <span style="color:red">46.4%</span> | 0.112 |
| Sokoban | Claude Sonnet 4.6 Low Thinking | 51.6% | 5.65 | 55.6% | 400 (215/185) | 97.2% | <span style="color:red">21.1%</span> | 45.1% | 0.148 |
| Sokoban | GPT-5.2 Instant | 35.2% | 9.02 | 35.4% | <span style="color:red">661 (189/472)</span> | 96.3% | 19.9% | 36.0% | <span style="color:red">0.167</span> |

Columns:
- `Estimations (Succ/Fail Rollouts)` means total estimation samples, followed by counts from successful and failed rollouts.
- `Pred Hit on Success/Fail Rollouts` is the mean of `can_finish_correct` over all estimation samples in that rollout group.
- `Success-Rollout Interval Hit` counts a success-rollout sample as a hit only when the predicted interval contains the actual remaining token target; missing intervals count as misses.
- `Success-Rollout Reward` uses the custom score requested in `figure-plan.md`, averaged over success-rollout estimation samples.
