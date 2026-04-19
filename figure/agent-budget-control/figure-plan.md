### Figure1 estimation accuracy in first turn

统计第一个turn estimate 的准确率（混淆矩阵）

### Figure2 estimation accuracy in all turns

横坐标是relative position(0-0.1/0.1-0.2/...) 这样排列
纵坐标是accuracy(是否预测准确)
画三条线：总准确率/预测正确的准确率/预测错误的准确率

### Figure3 reward curve


横坐标是relative position(0-0.1/0.1-0.2/...) 这样排列
纵坐标是 reward
画三条线：总reward/true rollout的reward/false rollout 的 reward

reward 规则：
Agent output:
 [\hat{R_{k(low)}},  \hat{R_{k(high)}}]. 
Score:
If real is success: 1(R_k in [\hat{R_{k(low)}},  \hat{R_{k(high)}}]) *max(0, (1 - (\hat{R_{k(high)}} -  \hat{R_{k(low)}})/R_k))
If real is fail: output(will fail in given budget), otherwise 0


### Figure4 hit rate and 乐观与悲观 in success rollout

横坐标是relative position(0-0.1/0.1-0.2/...) 这样排列
纵坐标是 0-1
柱状图叠加，每一个柱子有hit/过于乐观和过于悲观

### Figure5 range width 变化 in success rollouts

横坐标是relative position(0-0.1/0.1-0.2/...) 这样排列
纵坐标是 range width


### Figure6 cached tokens in both rollouts and estimations

### Figure7 average token used in each rollout turns

### Table1 in overall folder

画一个表
为每一个model 在 每一个benchmark 上面写
1. rollout 成功率
2. rollout 平均turn 数量
3. 首轮success 预测成功率
4. estimation 数量，其中estimation success rollout 有几个 faill rollout 有几个
4. 所有estimation中，fail/success rollout 的预测命中率
4. 在success rollout 中的预测命中率
5. 在success rollout 中的reward (按照Agent output:
 [\hat{R_{k(low)}},  \hat{R_{k(high)}}]. 
Score:
If real is success: 1(R_k in [\hat{R_{k(low)}},  \hat{R_{k(high)}}]) *max(0, (1 - (\hat{R_{k(high)}} -  \hat{R_{k(low)}})/R_k))
If real is fail: output(will fail in given budget), otherwise 0) 来计算

给每个benchmark的SOTA模型标红