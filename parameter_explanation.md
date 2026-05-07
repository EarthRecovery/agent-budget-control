你需要借助ragen搭建一套estimation 的评测环境
你的输入是类似于${HOME}/database/origin-test/warehouse-origin-gpt5.2instant-5-test/rollout_20260417_220923_seed46.json这样的json文档

接下来你需要创建一个临时文件，文件是一个json, 包含输入和输出

假如一个rollout 是：
"messages": [
          {
            "role": "system",
            "content": "You're a helpful assistant. You are a search agent answering questions by searching for information. Choose exactly one valid next action. Use search[<query>] to retrieve evidence and finish[<answer>] only when you are ready to submit the final answer. Inside <answer>, output exactly one action string only."
          },
          {
            "role": "user",
            "content": "\nTurn 1:\nState:\nQuestion: [PERSON], began his career in the 1970s and has written over how many major-label recorded songs including several number-one hits, another artist who have recorded his work include [PERSON], an American country music artist?\nAvailable actions: search[<query>], finish[<answer>]\nYou have 10 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Search for [PERSON] songwriter written over how many major-label recorded songs including number-one hits; verify number.</think><answer>search[[PERSON] songwriter written over how many major-label recorded songs including several number-one hits]</answer>"
          },
          {
            "role": "user",
            "content": "Reward:\n0.0\n\nTurn 2:\nState:\nSearch results for '[PERSON] songwriter written over how many major-label recorded songs including several number-one hits':\nCould not connect to retrieval server at http://127.0.0.1:8000. Is it running?\n\nAvailable actions: search[<query>], finish[<answer>]\nYou have 9 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Try a different query focusing on bio lines (“over X major label cuts”).</think><answer>search[\"[PERSON]\" \"major-label recorded songs\" over]</answer>"
          },
          {
            "role": "user",
            "content": "Reward:\n0.0\n\nTurn 3:\nState:\nSearch results for '\"[PERSON]\" \"major-label recorded songs\" over':\nCould not connect to retrieval server at http://127.0.0.1:8000. Is it running?\n\nAvailable actions: search[<query>], finish[<answer>]\nYou have 8 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Try one more query variant that may hit cached snippet; if still down, will finish with best-known value.</think><answer>search[[PERSON] songwriter over 200 major label recorded songs]</answer>"
          },

简化为：
system
user1
assistant1
user2
assistant2
user3
assistant3
...

你需要将其整理为：
{
  输入：user1
  输出：assistant1
}
{
  输入:user1,assistant1，user2
  输出：assistant2
}
'''

每一个rollout 都应该整理成多个这样的输入输出对

接下来你需要设计一个程序：

输入给agent api：system(新设计的) + 输入
|
agent api 处理
|
输出给处理程序

接下来处理程序处理输出，将输入和输出以及处理结果汇总到一个新的json上面去

这里有几个概念：
1. 花费的时间： 指从开始到结束用了多少个week, 在设定中是2个week 做一次决策
2. 仓库累计占用：指的是仓库累计占用的多少的空间.周，注意出仓并不会减少这个值，但是进仓会增加这个值，所以是累计占用
3. 金钱累计花销： 指的是一个花了多少钱，注意赚钱并不会减少这个值

我希望你能够将这个流程写为一个env,同时为这个env 能够定制一些配置
比如说你需要完成这样一个配置：
system输入为：你是一个evaluation 评估agent，你需要根据根据输入的信息评估：
1. 这个rollout能不能够在 {花费时间小于x周 + 仓库累计占用小于y件.周 + 金钱累计花销 小于 z元}下成功做完？如果不能做完就回复impossible
2. 如果能够成功做完，需要总共花多少{花费时间 + 仓库累计占用 + 金钱累计花销}来完成?（这里三个值都应该需要给出一个范围内）
你需要根据<think></think><answer></answer>这样的格式输出

下面是一个示例：
{
            "role": "system",
            "content": "You are the supply chain manager for Acme, a consumer electronics brand.\nEvery 2 weeks you receive a state report and must decide production orders and retailer replenishment.\n\nSKUs: JCD543 (USB-C Hub, MOQ 500), JCD612 (Docking Station, MOQ 300), JCA374 (HDMI Cable, MOQ 1000), JUA365 (Display Adapter, MOQ 500), JCD389 (Travel Dock, MOQ 400).\nRetailers: MegaMart, TechZone, ValueCo, OfficePlus, SD.\n\nAvailable commands (separate multiple with semicolons):\n  produce <SKU> <level:1-3> <ocean|air>   -- order level x MOQ units; level 2 = 8% discount, level 3 = 15% discount; ocean ~32 days, air ~6 days\n  replenish <Retailer> <SKU> <trucks:1-3>  -- send trucks from Acme warehouse to retailer\n  draw_credit <amount>                      -- borrow from $200K credit line (0.8% interest/step)\n  repay_credit <amount>                     -- repay outstanding credit balance\n  factor_ar <amount>                        -- sell accounts receivable at 5% discount for immediate cash\n  pass                                      -- take no action this turn\n\nYou MUST wrap your reasoning in <think>...</think> and your action in <answer>...</answer>.\n\nExample:\n<think>MegaMart is running low on JCD543 and demand was 300 last period. I should replenish and start a new production run.</think>\n<answer>produce JCD543 2 ocean; replenish MegaMart JCD543 2; replenish TechZone JCD612 1</answer>\n\nGoal: maximize cumulative profit while avoiding stockouts and bankruptcy."
          },
          {
            "role": "user",
            "content": "\nTurn 1:\nState:\n=== Step 0 (Day 0) ===\nCash: $500,000 | AR Outstanding: $0\nCredit: $0 used / $200,000 available\n\nAcme Warehouse (cap 50000):\n  JCD543 (USB-C 9-in-1 Multi-Port Hub): 12204 in stock, 0 in-transit, 0 in production\n  JCD612 (USB-C 4K Docking Station): 13557 in stock, 0 in-transit, 0 in production\n  JCA374 (USB-C to 4K HDMI Cable 1.8m): 12258 in stock, 0 in-transit, 0 in production\n  JUA365 (USB 3.0 to HDMI/VGA Adapter): 16252 in stock, 0 in-transit, 0 in production\n  JCD389 (USB-C Compact Travel Dock): 11077 in stock, 0 in-transit, 0 in production\n\nRetailers:\n  MegaMart (inv 5679/25000):\n    JCD543: 1402 stock, 0 in-transit, last demand 0\n    JCD612: 1394 stock, 0 in-transit, last demand 0\n    JCA374: 1063 stock, 0 in-transit, last demand 0\n    JUA365: 875 stock, 0 in-transit, last demand 0\n    JCD389: 945 stock, 0 in-transit, last demand 0\n  TechZone (inv 23652/18000):\n    JCD543: 4354 stock, 0 in-transit, last demand 0\n    JCD612: 5852 stock, 0 in-transit, last demand 0\n    JCA374: 5055 stock, 0 in-transit, last demand 0\n    JUA365: 4106 stock, 0 in-transit, last demand 0\n    JCD389: 4285 stock, 0 in-transit, last demand 0\n  ValueCo (inv 11680/15000):\n    JCD543: 1637 stock, 0 in-transit, last demand 0\n    JCD612: 2029 stock, 0 in-transit, last demand 0\n    JCA374: 759 stock, 0 in-transit, last demand 0\n    JUA365: 5662 stock, 0 in-transit, last demand 0\n    JCD389: 1593 stock, 0 in-transit, last demand 0\n  OfficePlus (inv 6829/8000):\n    JCD543: 1561 stock, 0 in-transit, last demand 0\n    JCD612: 745 stock, 0 in-transit, last demand 0\n    JCA374: 1209 stock, 0 in-transit, last demand 0\n    JUA365: 1536 stock, 0 in-transit, last demand 0\n    JCD389: 1778 stock, 0 in-transit, last demand 0\n  SD (inv 6604/7000):\n    JCD543: 1213 stock, 0 in-transit, last demand 0\n    JCD612: 1276 stock, 0 in-transit, last demand 0\n    JCA374: 2126 stock, 0 in-transit, last demand 0\n    JUA365: 1362 stock, 0 in-transit, last demand 0\n    JCD389: 627 stock, 0 in-transit, last demand 0\nYou have 11 actions left.\n"
          },
          {
            "role": "assistant",
            "content": "<think>It\u2019s Day 0 with very high inventory at Acme and retailers, and there is no demand history yet (all last demand = 0). Also, TechZone is already over its total inventory capacity (23,652/18,000), so replenishing there would likely be impossible/inefficient. With no immediate stockout risk and plenty of supply everywhere, I should avoid producing (to prevent tying up cash and adding holding risk) and wait for the first demand signal.</think><answer>pass</answer>"
          },
          {
            "role": "user",
            "content": "Reward:\n804077.7799999999\n\nTurn 2:\nState:\n=== Step 1 (Day 14) ===\nCash: $438,256 | AR Outstanding: $865,821\nCredit: $0 used / $200,000 available\n\nAcme Warehouse (cap 50000):\n  JCD543 (USB-C 9-in-1 Multi-Port Hub): 12204 in stock, 0 in-transit, 0 in production\n  JCD612 (USB-C 4K Docking Station): 13557 in stock, 0 in-transit, 0 in production\n  JCA374 (USB-C to 4K HDMI Cable 1.8m): 12258 in stock, 0 in-transit, 0 in production\n  JUA365 (USB 3.0 to HDMI/VGA Adapter): 16252 in stock, 0 in-transit, 0 in production\n  JCD389 (USB-C Compact Travel Dock): 11077 in stock, 0 in-transit, 0 in production\n\nRetailers:\n  MegaMart (inv 4237/25000):\n    JCD543: 1035 stock, 0 in-transit, demand [367]\n    JCD612: 977 stock, 0 in-transit, demand [417]\n    JCA374: 575 stock, 0 in-transit, demand [488]\n    JUA365: 705 stock, 0 in-transit, demand [170]\n    JCD389: 945 stock, 0 in-transit, demand [0]\n  TechZone (inv 14358/18000):\n    JCD543: 2853 stock, 0 in-transit, demand [1501]\n    JCD612: 3478 stock, 0 in-transit, demand [2374]\n    JCA374: 2840 stock, 0 in-transit, demand [2215]\n    JUA365: 2471 stock, 0 in-transit, demand [1635]\n    JCD389: 2716 stock, 0 in-transit, demand [1569]\n  ValueCo (inv 7023/15000):\n    JCD543: 944 stock, 0 in-transit, demand [693]\n    JCD612: 1174 stock, 0 in-transit, demand [855]\n    JCA374: 413 stock, 0 in-transit, demand [346]\n    JUA365: 3526 stock, 0 in-transit, demand [2136]\n    JCD389: 966 stock, 0 in-transit, demand [627]\n  OfficePlus (inv 4275/8000):\n    JCD543: 932 stock, 0 in-transit, demand [629]\n    JCD612: 454 stock, 0 in-transit, demand [291]\n    JCA374: 902 stock, 0 in-transit, demand [307]\n    JUA365: 919 stock, 0 in-transit, demand [617]\n    JCD389: 1068 stock, 0 in-transit, demand [710]\n  SD (inv 4089/7000):\n    JCD543: 690 stock, 0 in-transit, demand [523]\n    JCD612: 786 stock, 0 in-transit, demand [490]\n    JCA374: 1325 stock, 0 in-transit, demand [801]\n    JUA365: 825 stock, 0 in-transit, demand [537]\n    JCD389: 463 stock, 0 in-transit, demand [164]\n\nCash flow: $500,000 \u2192 $438,256\n  Holding costs: -$45,744\n  Fixed opex: -$16,000\nPeriod P&L: revenue $865,821, holding $45,744, stockout penalty $0, overstock penalty $0\nYou have 10 actions left.\n"
          }

{"role": "assistant",
 "content": "

Based on the provided rollout context, you are provided below information:
 1. You have completed {completed weeks} weeks in {completed_turns} turns
 2. Each 2 weeks, your consumption is {Day0: A,B,C;  Day14:A,B,C …}.
 3. You need to 将你的存款超过x元 within {花费时间小于x周 + 仓库累计占用小于y件.周 + 金钱累计花销 小于 z元}

Now, estimate:
1. Whether you can 将你的存款超过x元 successfully within {花费时间小于x周 + 仓库累计占用小于y件.周 + 金钱累计花销 小于 z元}.
2. If yes, how many {花费时间 + 仓库累计 + 金钱累计花销} are still needed to finish the task, starting from the next turn. Return an estimation interval: at least est_low and at most est_high
3. If no, answer "impossible".
4. You should try your best to estimate whether the task can finish within budget (most important). If you think the task can finish within budget, your interval should be as tight as possible while still covering the true remaining token budget.

E.g. For a three-turn interaction, suppose only Turn 1 has been completed.
The full interaction is: {Turn 1: input X1 tokens, output Y1 tokens; Turn 2: input X2 tokens, output Y2 tokens; Turn 3: input X3 tokens, output Y3 tokens}
You will receive:
{turn_token_usage_text: Turn 1: input X1 tokens, output Y1 tokens}
You should estimate:
{X2 + Y2 + X3 + Y3}


Output exactly one of the following:
<think>[YOUR THINKING]</think><answer>花费时间:[est_low, est_high],仓库累计占用:[est_low, est_high],金钱累计花销:[est_low, est_high]</answer>
or
<think>[YOUR THINKING]</think><answer>impossible</answer>
"
}

接下来处理程序需要处理这些输出，计算和实际的偏差值，将结果汇总到一个json文件里面

在prompt 中的所有语言都应该是英文，你需要根据${HOME}/database/origin-test/warehouse-origin-gpt5.2instant-5-test/rollout_20260417_220923_seed46.json 里面的记录选择合适的翻译

你需要把这些整理为一个环境：money_estimation. 很多逻辑和${HOME}/agent-budget-control/ragen/env/token_estimation 很像，你可以参照着来。做完以后你需要在${HOME}/agent-budget-control/scripts/evaluation-scripts/eval 下面写一个warehouse.sh