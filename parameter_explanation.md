# Parameter Explanation

## `agent_proxy.eval-estimation-single`

**作用**：启用“单次估计模式”。模型在每次回答前只需要预测“本次回答要用多少 token”，并把估计、上下文、输出和实际用量记录进评估日志。

### Context / Prompt 侧

- **FORMAT_PROMPT**：`ContextManager._build_format_prompt` 会要求输出
  `"<budget-thinking>...</budget-thinking><token_estimation>...</token_estimation><think>...<answer>..."`，
  不包含 `<turn_estimation>`。文件：`agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **生成前缀**：`LLMAgentProxy._get_generation_suffix` 与 `ContextManager._get_generation_prefix`
  在 single 模式下都返回 `"<budget-thinking>"`。文件：`agent-budget-control/ragen/llm_agent/agent_proxy.py`，
  `agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **用户提示注入**：`CtxManagerWrapper.intercept` 会在最后一条 user message 追加“先输出
  `<token_estimation>` 估计”的提示。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **记录输入**：`CtxManagerWrapper._record_estimation_inputs` 会保存 message/prompt/suffix 等信息，
  用于最终评估日志。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 输出解析与日志

- **强制 `<budget-thinking>` 前缀**：`CtxManagerWrapper._decorate_response_for_estimation` 在 single 模式下确保
  raw response 以 `<budget-thinking>` 开头。
- **估计值提取**：`CtxManagerWrapper._record_estimation_outputs` 提取 `<token_estimation>` 的整数值写入
  `estimate_token`，不会写 `estimate_remaining_turn`。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **日志落盘**：`CtxManagerWrapper.finalize_rollout` 会输出 `*_eval_estimation_dialogues.json`，
  每个 turn 只包含 token 估计相关字段。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### ES 侧分析与奖励

- `EsManagerWrapper._apply_token_estimation_adjustment` 读取 `<token_estimation>`，
  计算 `estimate_success`、`estimate_token_diff`、`estimate_token_error_ratio` 等，
  并按 `token_estimation_reward` 调整 reward。文件：`agent-budget-control/ragen/wrapper/es_manager_wrapper.py`.
- `EsManager.get_rollout_states` 汇总 `token_estimation_success_rate`、
  `token_estimation_missing_tag_rate`、`token_estimation_mean_abs_error` 等指标。
  文件：`agent-budget-control/ragen/llm_agent/es_manager.py`.


## `agent_proxy.eval-estimation-multi`

**作用**：启用“多次估计模式”。模型在每次回答前需要预测两件事：
1) **剩余还需要多少 turn**（含当前 turn），2) **本 turn 预计使用多少 token**。并记录到评估日志中。

### Context / Prompt 侧

- **FORMAT_PROMPT**：`ContextManager._build_format_prompt` 要求输出
  `"<budget-thinking>...</budget-thinking><turn_estimation>...</turn_estimation><token_estimation>...</token_estimation><think>...<answer>..."`。
  文件：`agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **生成前缀**：同样返回 `"<budget-thinking>"`。文件：`agent-budget-control/ragen/llm_agent/agent_proxy.py`，
  `agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **用户提示注入**：`CtxManagerWrapper.intercept` 会在最后一条 user message 追加两问提示，
  让模型先给 `<turn_estimation>`、`<token_estimation>`。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **记录输入**：与 single 相同，输入上下文会被记录到估计日志缓存。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 输出解析与日志

- **估计值提取**：`CtxManagerWrapper._record_estimation_outputs` 会提取
  `<turn_estimation>` 写入 `estimate_remaining_turn`，同时提取 `<token_estimation>` 写入 `estimate_token`。
  文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **rollout 结束后追加真实剩余 turn**：`CtxManagerWrapper.finalize_rollout` 会计算
  `actual_remaining_turn = 总 reward turn 数 - 当前 turn + 1`，并写入日志。文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### ES 侧分析与奖励

- token 估计部分的 reward 调整与统计逻辑与 single 相同（读取 `<token_estimation>`）。
- 若 `benchmark_factors.mode=turn`，`EsManagerWrapper.compute_benchmark_factors` 会读取
  `<turn_estimation>` 做 turn 估计统计（例如 `turn_estimate_mean_abs_error` 等）。文件：
  `agent-budget-control/ragen/wrapper/es_manager_wrapper.py`.


## `agent_proxy.eval-estimation-toolcall`

**默认值**：`false`

**作用**：启用“action-point 估计模式”。这个模式只适用于 **Robotouille**，模型在每次回答前需要先预测两件事：
1) **从当前 turn 开始，到任务结束一共还需要多少 action points**（包含当前 turn），
2) **本 turn 预计会消耗多少 action points**。

这些估计会和真实 action-point 用量一起写入 `*_eval_estimation_dialogues.json`。

### 运行前校验

- **只允许在 Robotouille 上开启**：`agent_proxy.eval-estimation-toolcall=true` 时，当前 active env 的
  `env_type` 必须全部是 `robotouille`；如果混入其他环境，会直接报错。文件：
  `agent-budget-control/ragen/llm_agent/eval_config.py`，
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **必须同时开启 action budget**：对应的
  `custom_envs.<Tag>.env_config.enable_action_budget` 必须是 `true`，否则直接报错。文件：
  `agent-budget-control/ragen/llm_agent/eval_config.py`.
- **必须有 `max_action_points`，并用它做截断**：代码会读取 Robotouille 的
  `max_action_points` 作为 action-point 估计上界；模型输出的两个估计值都会被裁到
  `[0, max_action_points]`。如果 active Robotouille env 的 `max_action_points` 不一致，当前实现会直接报错。文件：
  `agent-budget-control/ragen/llm_agent/eval_config.py`，
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### Context / Prompt 侧

- **FORMAT_PROMPT**：`ContextManager._build_format_prompt` 会要求输出
  `"<budget-thinking>...</budget-thinking><remaining_action_points_estimation>...</remaining_action_points_estimation><action_points_estimation>...</action_points_estimation><think>...<answer>..."`。
  文件：`agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **生成前缀**：`LLMAgentProxy._get_generation_suffix` 与 `ContextManager._get_generation_prefix`
  在 toolcall 模式下同样返回 `"<budget-thinking>"`。文件：
  `agent-budget-control/ragen/llm_agent/agent_proxy.py`，
  `agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **用户提示注入**：`CtxManagerWrapper.intercept` 会在最后一条 user message 追加 action-point 版双问题提示，
  并明确要求填写 `<remaining_action_points_estimation>` 与 `<action_points_estimation>`；如果存在
  `max_action_points`，提示里也会强调整数范围。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 输出解析与日志

- **估计值提取**：`CtxManagerWrapper._record_estimation_outputs` 与
  `CtxManagerWrapper.finalize_rollout` 会提取：
  - `<remaining_action_points_estimation>` -> `estimate_remaining_action_points`
  - `<action_points_estimation>` -> `estimate_action_points`
  这两个值都会按 `max_action_points` 做截断。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **真实本 turn action-point 用量**：`EnvStateManager.step` 会把一整个 turn 内真正消耗的
  action points 聚合成 `action_points_used`，而不是只保留最后一个 action 的 cost。文件：
  `agent-budget-control/ragen/llm_agent/es_manager.py`.
- **rollout 结束后追加真实剩余 action points**：`CtxManagerWrapper.finalize_rollout` 会基于
  每个 reward turn 的 `action_points_used`，计算：
  - `actual_action_points`：本 turn 实际消耗多少 action points；
  - `actual_remaining_action_points`：从当前 turn 开始直到结束，一共实际消耗多少 action points。
  同时日志里也会记录 env 级别的 `max_action_points`。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 与 Robotouille action budget 的关系

- 这个模式本身**不会替代** Robotouille 的 action budget 机制；真正的 budget 扣减和终止逻辑仍由
  `enable_action_budget` / `max_action_points` 控制。文件：
  `agent-budget-control/ragen/env/robotouille/env.py`.
- `eval-estimation-toolcall` 做的是“让模型先估计，再把估计和真实 action-point 用量写进评估日志”，
  不是额外再造一套 budget 执行器。

## `agent_proxy.eval_compliance_token`

**作用**：启用“token budget compliance”评测模式。这个模式**不是**让模型先输出
`<token_estimation>` / `<turn_estimation>`，而是把一组 token budget 展开成多份 rollout，
让同一个原始 group / 题目复制出多份 env，每份 env 绑定一个固定的 token 上限；然后在
rollout 结束后统计该 env 是否在对应上限内完成回答。

### 运行前校验

- **与估计模式互斥**：`agent_proxy.eval_compliance_token` 不能和
  `agent_proxy.eval-estimation-single`、`agent_proxy.eval-estimation-multi` 同时开启。
  否则会在 `CtxManagerWrapper._validate_eval_modes` 中直接报错。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **必须提供 scope**：如果该参数为 `true`，则 `agent_proxy.eval_compliance_token_scope`
  不能为空；否则同样会直接报错。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`，
  `agent-budget-control/ragen/llm_agent/eval_config.py`.

### Rollout / Context 侧

- **运行时会扩展 `group_size`，而不是覆盖 `max_turn`**：
  `expand_compliance_group_size` 会在 `LLMAgentProxy` 初始化时，把
  `es_manager.train.group_size` 和 `es_manager.val.group_size` 乘上
  `len(agent_proxy.eval_compliance_token_scope)`；`agent_proxy.max_turn` 保持原值不变。文件：
  `agent-budget-control/ragen/llm_agent/eval_config.py`，
  `agent-budget-control/ragen/llm_agent/agent_proxy.py`.
- **同一个原始 group / 题目会被复制成多份 rollout**：扩展后，同一个 `group_id` 内的 env
  会继续共享同一个 seed，因此仍对应同一道题；只是它们会被按位置分配不同的
  `compliance_token_limit`。如果原始 `group_size=1`，那就是每个题复制
  `len(scope)` 份；如果原始 `group_size=k`，那每个 budget 会连续重复 `k` 份。文件：
  `agent-budget-control/ragen/llm_agent/es_manager.py`，
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **每个 env 绑定一个固定 budget，不是按 turn 取 `scope[i]`**：
  `CtxManagerWrapper._get_eval_compliance_token_limit_for_env` 会根据当前 env 在扩展后 group
  里的位置，给这个 env 分配一个固定的 `compliance_token_limit`。该 env 后续所有 turn 注入的
  都是同一个 budget。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **每个 turn 都会注入对应 env 的 token 限制提示**：`CtxManagerWrapper.intercept` 会在最后一条
  user message 追加一句：`"You must finish your answer in N tokens."`，其中 `N` 是当前 env
  绑定的固定 budget。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **默认回答前缀回到原生格式**：当 `eval-estimation-single=False` 且
  `eval-estimation-multi=False` 时，生成前缀不再默认使用 `<budget-thinking>`，而是恢复为
  `enable_think=True` 时从 `<think>` 开始、否则从 `<answer>` 开始。也就是说，compliance
  模式本身不会自动引入 `<budget-thinking>`。文件：
  `agent-budget-control/ragen/llm_agent/agent_proxy.py`，
  `agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **不再把 `Max response length: ...` 写入 context**：当 compliance 模式开启时，
  `ContextManager._build_format_prompt` 会把 `LENGTH_PROMPT` 置空，因此 context 中不会再出现
  `Max response length: 1024 words (tokens).` 这类文案。文件：
  `agent-budget-control/ragen/llm_agent/ctx_manager.py`.
- **但底层真实生成上限仍然存在**：这一模式目前是**软合规（soft compliance）**，不是硬截断。
  模型仍然受全局 `actor_rollout_ref.rollout.response_length`（以及 API 侧
  `max_tokens` / `max_completion_tokens`）约束；`eval_compliance_token_scope`
  不会被下发成每个 env 的真实 `max_tokens`。如果模型超过该 env 的目标 token 数，生成依然会继续，
  只是在日志里被记为不合规。

### 输出解析与日志

- **日志文件名**：该模式会输出 `*_eval_compliance_dialogues.json`，不同于估计模式的
  `*_eval_estimation_dialogues.json`。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **rollout 元信息**：`LLMAgentProxy.rollout` 会把日志路径写入
  `rollouts.meta_info["eval_compliance_json_path"]`。文件：
  `agent-budget-control/ragen/llm_agent/agent_proxy.py`.
- **JSON 顶层是展开后的 env rollout 记录**：也就是说，在 compliance 模式下，
  顶层记录数会按扩展后的 `env_groups * group_size` 计算。每条顶层记录对应一个 env rollout，
  下面再挂这个 env 自己的 `turns` 列表。
- **env 级字段**：日志中会记录 `mode="compliance_token"`、完整的
  `eval_compliance_token_scope`，以及该 env 绑定的 `compliance_token_limit`。文件：
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.
- **turn 级字段**：每轮会记录：
  - `compliance_token_limit`：当前 env 绑定的 token 上限；
  - `compliance_instruction`：注入到 prompt 的提示语；
  - `actual_token`：该轮实际生成 token 数；
  - `has_answer`：该轮是否成功产生了回答；
  - `within_token_limit`：`actual_token <= compliance_token_limit`；
  - `answered_within_token_limit`：是否“既有回答、又在上限内”，这是最直接的合规字段；
  - `token_limit_delta`：`actual_token - compliance_token_limit`，正数表示超了多少。
  文件：`agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 结果解释上的注意事项

- **`max_turn` 语义没变**：scope 不再决定 rollout 最多跑几轮；每个复制出来的 env 仍然按
  `agent_proxy.max_turn` 的逻辑继续 rollout。
- **scope 决定的是“每个原始题目要复制多少份 budget rollout”**：
  比如 `VAL_GROUPS=2`、原始 `group_size=1`、`eval_compliance_token_scope=[100,200,300,400,500]`，
  那运行时会展开成 **10 个 env rollout**，JSON 顶层也会有 **10 条 env 记录**。
  其中每个原始题目对应 5 条记录，token budget 分别是 100/200/300/400/500。
- **推荐看 `answered_within_token_limit`**：`within_token_limit=True` 只表示 token 数没超；
  若生成报错或没有真正回答，最终仍可能不是一次有效合规完成。更稳妥的字段是
  `answered_within_token_limit`。


## `agent_proxy.eval_compliance_token_scope`

**作用**：提供 compliance 模式下的“budget 列表”。这个列表不是给单个 rollout 按 turn 轮流使用，
而是用来把每个原始 group / 题目展开成多份 env rollout，每份 env rollout 绑定其中一个 budget。

### 参数含义

- **类型**：列表，例如 `[]`、`[100, 200, 300]`。
- **每个元素的含义**：第 `i` 个元素表示展开后的第 `i` 组 env 副本所绑定的 token 上限。
- **运行时会被规范化为非负整数**：代码会尝试把每个元素转成 `int`，并裁到 `>=0`。
  如果某个元素无法转成整数，会直接报错。文件：
  `agent-budget-control/ragen/llm_agent/eval_config.py`，
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py`.

### 例子

- `agent_proxy.eval_compliance_token_scope=[100,200,300,400,500]`
  表示：
  - 每个原始题目会复制出 5 份 env rollout；
  - 这 5 份 rollout 分别绑定 100、200、300、400、500 这 5 个 token budget；
  - 每份 rollout 在自己的所有 turn 中，都会注入同一句固定提示：
    `You must finish your answer in N tokens.`
  - 如果原始配置是 `VAL_GROUPS=2`、原始 `group_size=1`，那么运行时会展开成
    2 × 5 = 10 个 env rollout，JSON 顶层同样会有 10 条记录；
  - 如果原始 `group_size=2`，那么每个 budget 会连续重复 2 份，因为同一个原始 group
    里原本就有 2 个并行副本。

### 与其他参数的关系

- **只有在 `agent_proxy.eval_compliance_token=true` 时才生效**。
- **它会扩展 `group_size`，不会覆盖 `agent_proxy.max_turn`**。
- **不会覆盖真实生成上限**：它只控制“每个 env rollout 希望模型遵守的 token 预算”和日志统计，
  不会替代 `actor_rollout_ref.rollout.response_length` 的硬上限。

## `agent_proxy.eval_compliance_turn`

**作用**：启用“turn budget compliance”评测模式。整体展开方式应与
`agent_proxy.eval_compliance_token` 保持一致：系统先读取
`agent_proxy.eval_compliance_turn_scope`，再把每个原始 group / 题目复制成多份 env rollout，
每份 env rollout 绑定一个固定的 `budget_turn_num`（建议日志字段名使用
`compliance_turn_limit`），分别执行一整条 multi-turn rollout。区别在于，这里评估的不是
“单轮生成 token 是否超限”，而是“整条 rollout 最终一共用了多少 turn，是否在 budget 内”。

### 运行前校验

- **与估计模式互斥**：`agent_proxy.eval_compliance_turn` 不能和
  `agent_proxy.eval-estimation-single`、`agent_proxy.eval-estimation-multi` 同时开启。
- **必须提供 scope**：如果该参数为 `true`，则
  `agent_proxy.eval_compliance_turn_scope` 不能为空；否则应直接报错。
- **建议与 token compliance 互斥**：`eval_compliance_turn` 与
  `eval_compliance_token` 都会扩展 `group_size` 并注入 compliance prompt；如果要同时做两种
  budget 评测，需要单独定义笛卡尔积展开逻辑。当前更稳妥的要求是二者互斥。
- **scope 解析逻辑应与 token 版一致**：需要从
  `agent_proxy.eval_compliance_turn_scope` 中把每个元素转成 `int`，并裁到 `>=0`。
  若某个元素无法转成整数，应直接报错。建议在
  `agent-budget-control/ragen/llm_agent/eval_config.py` 和
  `agent-budget-control/ragen/wrapper/ctx_manager_wrapper.py` 中新增 turn 版的
  `resolve_eval_compliance_turn_scope` / `_get_eval_compliance_turn_scope`。

### Rollout / Context 侧

- **运行时会扩展 `group_size`，而不是覆盖 `max_turn`**：逻辑应与 token 版一致，
  `es_manager.train.group_size` 和 `es_manager.val.group_size` 按
  `len(agent_proxy.eval_compliance_turn_scope)` 扩大；`agent_proxy.max_turn` 保持原值不变。
- **同一个原始 group / 题目会被复制成多份 rollout**：每个复制出来的 env 绑定一个固定的
  `budget_turn_num`，并针对这个 budget 独立完成一次完整 rollout。
- **每个 env 绑定一个固定 turn budget，不是按 turn 轮流取 `scope[i]`**：也就是说，
  `scope` 的作用和 token 版完全一样，是“展开多少份 budget rollout”，而不是“单条 rollout
  在第 1/2/3 轮分别使用哪个 budget”。
- **每个 turn 都要动态注入 turn budget 提示**：在最后一条 user message 中追加当前 env
  的 turn budget 信息，至少要包含：
  - 固定的 `budget_turn_num`；
  - 当前是第几轮 `current_turn`；
  - 你距离 budget 还有多少轮，建议定义为
    `turn_budget_distance = budget_turn_num - current_turn`。
- **不同区间的提示文案应区分开**：
  - 当 `turn_budget_distance > 0` 时，应明确说明“距离 budget 还剩 N turn(s)”；
  - 当 `turn_budget_distance == 0` 时，应明确说明“当前 turn 就是最后一个 budgeted turn”；
  - 当 `turn_budget_distance < 0` 时，应明确说明“已经超出 budget `abs(turn_budget_distance)` turn(s)”。
- **超 budget 后仍要继续在 context 中保留提醒**：一旦
  `current_turn > budget_turn_num`，后续每一轮的 context 都要继续提示已经超出多少 turn，
  不能只在第一次超出时提醒一次。
- **这是 soft compliance，不是 hard early stop**：即使某个 env 已经超出 turn budget，
  rollout 仍然继续跑到任务结束或 `agent_proxy.max_turn`。超 budget 只影响 prompt 提示和日志统计，
  不应直接截断生成。
- **建议不要直接复用现有 `budget_turn` 字段做日志主字段**：仓库里已有 `budget_turn`
  用于 mixed budget / benchmark 相关逻辑。为了避免和 turn compliance 混淆，建议 turn
  compliance 单独记录 `compliance_turn_limit`。
- **生成前缀保持原生格式**：和 token compliance 一样，turn compliance 本身不要求模型输出
  `<budget-thinking>`、`<turn_estimation>` 等额外标签；回答前缀应回到正常的
  `<think>` / `<answer>` 逻辑。

### 输出解析与日志

- **日志文件名建议继续复用**：仍然写入 `*_eval_compliance_dialogues.json`，但 env 级
  `mode` 应区分成 `compliance_turn`，避免和 `compliance_token` 混淆。
- **JSON 顶层仍是展开后的 env rollout 记录**：也就是说，顶层记录数仍按
  `env_groups * group_size * len(eval_compliance_turn_scope)` 计算。
- **env 级字段应至少包含**：
  - `mode="compliance_turn"`；
  - 完整的 `eval_compliance_turn_scope`；
  - 当前 env 绑定的 `compliance_turn_limit`；
  - `total_turns`：该 rollout 最终实际用了多少 reward turn；
  - `within_turn_limit`：`total_turns <= compliance_turn_limit`；
  - `turn_limit_delta`：`total_turns - compliance_turn_limit`，正数表示最终超了多少 turn；
  - `success_within_turn_limit`：任务成功且 `within_turn_limit=True`。
- **turn 级字段应至少包含**：
  - `compliance_turn_limit`：当前 env 绑定的固定 turn budget；
  - `current_turn`：当前是第几轮；
  - `turn_budget_distance`：`compliance_turn_limit - current_turn`；
  - `within_turn_limit_so_far`：`current_turn <= compliance_turn_limit`；
  - `exceeded_turn_limit`：`current_turn > compliance_turn_limit`；
  - `compliance_instruction`：实际注入到 prompt 的提醒文案。
- **最终统计以完整 rollout 为准**：每个 env rollout 结束后，都要在 env 级记录里写清楚
  “总共花了多少 turn”以及“是否在 budget 里面”；也就是用 `total_turns`、
  `within_turn_limit`、`turn_limit_delta` 做最终判定。

### 结果解释上的注意事项

- **`max_turn` 语义没变**：`eval_compliance_turn_scope` 决定的是“每个原始题目要复制多少份
  turn-budget rollout”，不是 rollout 最多跑多少轮。
- **超出 budget 不等于立即终止**：`within_turn_limit=False` 只表示最终 turn 数超标；
  rollout 本身仍可能继续生成直到 env 正常结束或达到 `max_turn`。
- **推荐优先看 `success_within_turn_limit`**：`within_turn_limit=True` 只说明最终 turn 数没超；
  若任务没有完成，不能算一次有效的 budget-compliant 成功。
- **每轮 distance 是相对“当前轮次”计算的**：推荐统一用
  `turn_budget_distance = compliance_turn_limit - current_turn`。正数表示还没到 budget，
  0 表示当前轮就是最后一个 budgeted turn，负数表示已经超出。


## `agent_proxy.eval_compliance_turn_scope`

**作用**：提供 turn compliance 模式下的“budget turn 列表”。这个列表不是给单个 rollout
按 turn 轮流使用，而是用来把每个原始 group / 题目展开成多份 env rollout，每份 env rollout
绑定其中一个固定的 `budget_turn_num`。

### 参数含义

- **类型**：列表，例如 `[]`、`[1, 2, 3]`。
- **每个元素的含义**：第 `i` 个元素表示展开后的第 `i` 组 env 副本所绑定的 turn 上限。
- **运行时应被规范化为非负整数**：逻辑与 `agent_proxy.eval_compliance_token_scope` 相同，
  需要尝试把每个元素转成 `int`，并裁到 `>=0`；若无法转成整数，应直接报错。
- **建议 budget 范围尽量落在 `[1, agent_proxy.max_turn]` 内**：大于 `max_turn` 虽然技术上可行，
  但评测意义会变弱；`0` 也可以被解析，但会导致从第一轮开始就处于“已超 budget”状态，
  通常没有实际价值。

### 例子

- `agent_proxy.eval_compliance_turn_scope=[1,2,3,4,5]`
  表示：
  - 每个原始题目会复制出 5 份 env rollout；
  - 这 5 份 rollout 分别绑定 1、2、3、4、5 这 5 个 turn budget；
  - 每份 rollout 的所有 turn 都会持续注入 turn budget 提示，但提示内容会随当前轮次动态变化；
  - 例如 budget=3 的那份 rollout：
    - turn 1 时提示“budget 是 3，距离 budget 还有 2 turn(s)”；
    - turn 3 时提示“当前就是最后一个 budgeted turn”；
    - turn 4 时提示“已经超出 budget 1 turn(s)”；
  - 如果原始配置是 `VAL_GROUPS=2`、原始 `group_size=1`，那么运行时会展开成
    2 × 5 = 10 个 env rollout，JSON 顶层也应有 10 条 env 记录；
  - 如果原始 `group_size=2`，那么每个 budget 会连续重复 2 份，因为同一个原始 group
    里原本就有 2 个并行副本。

### 与其他参数的关系

- **只有在 `agent_proxy.eval_compliance_turn=true` 时才生效**。
- **它会扩展 `group_size`，不会覆盖 `agent_proxy.max_turn`**。
- **不会变成真实的硬停止条件**：它只控制每个 env rollout 绑定的目标 turn budget、
  每轮 context 提示和最终日志统计，不会替代 `agent_proxy.max_turn` 的真实停止逻辑。
- **建议单独使用，不要和 `agent_proxy.eval_compliance_token` 同时开启**：否则需要额外定义
  token budget 与 turn budget 的联合展开规则。
