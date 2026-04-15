你需要借助ragen搭建一套estimation 的评测环境
你的输入是类似于/u/ylin30/database/origin-test/searchr1-origin-gpt5.2-instant-4-test/search_r1_api_eval_estimation_eval_estimation_dialogues.json 这样的json文档

接下来你需要创建一个临时文件，文件是一个json, 包含输入和输出

假如一个rollout 是：
"messages": [
          {
            "role": "system",
            "content": "You're a helpful assistant. You are a search agent answering questions by searching for information. Choose exactly one valid next action. Use search[<query>] to retrieve evidence and finish[<answer>] only when you are ready to submit the final answer. Inside <answer>, output exactly one action string only."
          },
          {
            "role": "user",
            "content": "\nTurn 1:\nState:\nQuestion: Gary Harrison, began his career in the 1970s and has written over how many major-label recorded songs including several number-one hits, another artist who have recorded his work include Bryan White, an American country music artist?\nAvailable actions: search[<query>], finish[<answer>]\nYou have 10 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Search for Gary Harrison songwriter written over how many major-label recorded songs including number-one hits; verify number.</think><answer>search[Gary Harrison songwriter written over how many major-label recorded songs including several number-one hits]</answer>"
          },
          {
            "role": "user",
            "content": "Reward:\n0.0\n\nTurn 2:\nState:\nSearch results for 'Gary Harrison songwriter written over how many major-label recorded songs including several number-one hits':\nCould not connect to retrieval server at http://127.0.0.1:8000. Is it running?\n\nAvailable actions: search[<query>], finish[<answer>]\nYou have 9 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Try a different query focusing on bio lines (“over X major label cuts”).</think><answer>search[\"Gary Harrison\" \"major-label recorded songs\" over]</answer>"
          },
          {
            "role": "user",
            "content": "Reward:\n0.0\n\nTurn 3:\nState:\nSearch results for '\"Gary Harrison\" \"major-label recorded songs\" over':\nCould not connect to retrieval server at http://127.0.0.1:8000. Is it running?\n\nAvailable actions: search[<query>], finish[<answer>]\nYou have 8 actions left. Always output: <budget-thinking> [Your budget-related reasoning] </budget-thinking> <turn_estimation> [Your estimated remaining turn count] </turn_estimation> <token_estimation> [Your estimated token count] </token_estimation> <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 2048 words (tokens).\n"
          },
          {
            "role": "assistant",
            "content": "<think>Try one more query variant that may hit cached snippet; if still down, will finish with best-known value.</think><answer>search[Gary Harrison songwriter over 200 major label recorded songs]</answer>"
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

我希望你能够将这个流程写为一个env,同时为这个env 能够定制一些配置
比如说你需要完成这样一个配置：
system输入为：你是一个evaluation 评估agent，你需要根据根据输入的信息评估：
1. 这个rollout能不能够在{max_turn} 下成功做完？
2. 如果能够成功做完，需要总共花多少token来完成?（这里应该需要给出一个范围，范围需要控制在200tokens以内）
3. 你预计下一个turn 需要花多少token?
你需要根据<think></think><can_finish></can_finish><remaining_turn></remaining_turn><next_turn_token_usage></next_turn_token_usage> 这样的格式输出

接下来处理程序需要处理这些输出，计算和实际的偏差值，将结果汇总到一个json文件里面