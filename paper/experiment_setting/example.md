4.1. Experimental Testbed
WeadopttheRAGEN[60]testbedandevaluate LLMagents onfour controllable tasks that stress
complementary decision-making regimes: irreversible planning (Sokoban), sparse-reward long
horizon navigation under stochastic transitions (FrozenLake), and symbolic math reasoning
(MetaMathQA, Countdown). To further evaluate multi-turn reasoning and decision-making
capabilities, we also include SearchQA [51], WebShop [64], and DeepCoder [25, 21, 15] (see
Appendix B.1 for detailed descriptions).
Environments and tasks. Our testbed spans seven diverse environments with complementary
characteristics (Table 3). Sokoban is a grid puzzle where the agent pushes boxes onto target cells;
actions are effectively irreversible since boxes cannot be pulled [39]. FrozenLake is a navigation
task with sparse rewards and stochastic transitions (slippery dynamics) [2]. MetaMathQA is a
mathQAtaskderivedfromMetaMathQA[67]wheretheagentmayreviseanswersovermultiple
attempts, and we apply a diminishing reward across retries (halving each retry). Countdown is
a single-turn numbers game [16] where the agent constructs an arithmetic expression to hit a
target. SearchQA is a multi-turn question-answering task where the agent iteratively searches
and synthesizes information to answer complex queries [51]. WebShop is an interactive web
navigation task where the agent must search and purchase products matching user specifications
[64]. DeepCoder is a code synthesis challenge where the agent generates program solutions to
meet specified input-output requirements [25, 21, 15].
Training and evaluation setup. We train Qwen2.5-3B [35] with the veRL/HybridFlow stack
[45], following RAGEN [60] defaults unless otherwise stated. We compare PPO [42], DAPO [68],
GRPO[44], and Dr. GRPO [23] for up to 400 rollout–update iterations. Each iteration collects
𝐾 =𝑃×𝐺 =128trajectories per environment, with prompt batch size 𝑃 = 8 and group size𝐺 = 16
trajectories per prompt. When applying SNR-Aware Filtering with keep rate 𝜌, we reduce the
effective minibatch size accordingly and scale the per-step loss by 𝜌, so the optimization step
size remains comparable.

Detailed Experimental Settings
B.1. Environments and Tasks
Weconstructadiverseseven-environmenttestbedtoevaluateLLMagentsacrosscomplementary
axes of decision-making complexity, including planning under irreversible dynamics (Sokoban),
long-horizon control with non-deterministic transitions (FrozenLake), multi-step symbolic
reasoning in mathematics (MetaMathQA, Countdown), multi-turn search and information
synthesis (SearchQA), goal-directed web navigation (WebShop), and program synthesis from
input-output specifications (DeepCoder). All environments are synthetic and fully controllable,
enabling clean analysis of RL learning from scratch without relying on real-world priors.
Sokoban. Weusethepuzzle Sokoban [39] to study multi-turn agent interaction with irreversible
dynamics. The agent must push boxes to designated target locations within a grid-based
warehouse. Unlike standard navigation tasks, Sokoban is characterized by irreversibility: boxes
can only be pushed, not pulled, meaning a single misstep can create unsolvable dead-ends
where boxes become permanently stuck against walls or corners. This requires the agent to
reason ahead and plan multi-step sequences before committing to actions. The reward signal
encourages both efficiency and accuracy: +1 for each box successfully placed on a target, −1
for moving a box off a target, +10 upon task completion, and −0.1 per action as a step penalty.
27
Weuseprocedurally generated puzzles with configurable room dimensions and box counts to
ensure diverse training scenarios.
Frozen Lake. This environment of FrozenLake [2] combines long-horizon decision-making
with deterministic transitions. The agent navigates a grid of frozen tiles to reach a goal while
avoiding holes that terminate the episode. We use the 2% random rate variant of Frozen Lake,
where each intended action is executed at a 98% probability. Rewards are sparse: only successful
goal-reaching trials receive a reward of +1, with all other outcomes yielding 0. The combination
of sparse rewards and long-horizon planning makes this environment challenging for credit
assignment.
MetaMathQA.Toevaluatemathematicalreasoningcapabilities, we include MetaMathQA [67], a
question-answering task drawn from the MetaMathQA dataset. Each episode presents the agent
with a mathematical problem requiring multi-step reasoning—ranging from arithmetic and
algebra to word problems and geometry. The agent must produce a final answer, and correctness
is determinedbyexactmatchwiththegroundtruth. Toencourageefficientreasoning, weemploy
a diminishing reward scheme: correct answers on the first attempt receive full reward (1.0), with
rewards halving for each subsequent attempt (0.5, 0.25, ...).
Countdown. Inspired by the numbers game from the TV show “Countdown” [16], this environ
ment tests compositional arithmetic reasoning. The agent is given a target number and a set
of source numbers, and must construct an arithmetic expression using each source number at
most once to reach the target exactly. For example, given target 24 and numbers [1,5,6,7], a
valid solution is 6× (7−5+1) +6. Rewards distinguish between format correctness and solution
correctness: full reward (1.0) for correct solutions, partial reward (0.1) for expressions that use
the correct numbers but yield incorrect results, and zero for malformed expressions.
DeepCoder. To evaluate agent capabilities in coding environments, we use DeepCoder, a coding
benchmark consisting of competitive programming problems. It was used to train DeepSeek
R1-Distill-Qwen-14B with reinforcement learning. The benchmark draws from three resources:
PrimeIntellect [25], TACO[21], and LiveCodeBench v5 (LCBv5) [15]. In this environment, agents
are required to generate a Python function that solves the given programming problem and
passes all hidden and public test cases. During training, rewards are assigned based on the
number of test cases successfully passed.
SearchQA. To evaluate multi-turn search and question-answering capabilities, we include
SearchQA from the RLLM framework [51], specifically the Search R1 variant. This environment
requires the agent to perform iterative web search and reasoning to answer open-domain
questions. The agent must formulate search queries, extract relevant information from retrieved
documents, and synthesize answers across multiple interaction turns. Rewards are based on
answer correctness and search efficiency, encouraging the agent to balance exploration breadth
with reasoning depth.
WebShop. WeuseWebShop[64], an interactive e-commerce environment for evaluating goal
directed multi-turn decision-making. The agent is presented with a shopping instruction (e.g.,
“find a red shirt under $30”) and must navigate a simulated online shopping website by issuing
search queries, clicking on products, and selecting appropriate items. The environment features
a large action space with realistic product catalogs and requires the agent to perform language
understanding, attribute matching, and sequential decision-making. Rewards are assigned
based on how well the purchased item matches the specified attributes and constraints.
28
B.2. Training and Evaluation Setup
We conduct our main experiments using Qwen2.5-3B and train with four policy-gradient
variants—PPO, DAPO, GRPO, and Dr.GRPO—for up to 400 rollout–update iterations on
NVIDIA GPUs using the veRL framework, with early stopping enabled as described below.
Each iteration collects 𝐾 = 128 trajectories per environment, organized as 𝑃 = 8 prompt groups
with 𝐺 =16parallel samples per prompt.
Episode horizons. To match task structure, the interactive environments (Sokoban, Frozen
Lake) use up to 5 interaction turns with 2 actions per turn (10 total actions per trajectory). The
single-step reasoning tasks (Countdown, MetaMathQA) use 1 turn with 1 action.
Optimization. We use an update batch size of 32 and a per-GPU minibatch size of 4. Policy
optimization uses GAE with (𝛾,𝜆) = (1.0,1.0) and Adam with (𝛽1, 𝛽2) = (0.9,0.999). The actor
learning rate is 1 ×10−6 and the critic learning rate is 1×10−5. We apply entropy regularization
with coefficient 𝛽 = 0.001. For PPO-based methods, we use asymmetric clipping with 𝜖low = 0.2
and 𝜖high = 0.28. We additionally impose a format penalty of −0.1 when the agent fails to output
a valid structured response (e.g., missing <think> or <answer> tags).
Early stopping. We stop training if either (i) reward-variance collapse is detected—the reward
variance drops below 10% of the baseline variance (defined as the mean variance over the first
10 training iterations) for 5 consecutive iterations—or (ii) the validation success rate remains
below 1% for 5 consecutive evaluation checkpoints.
Filtering ablation. We compare filtered rollouts with top_p = 0.9 (keeping the top 90% of
trajectory groups ranked by reward variance) against an unfiltered setting.
Evaluation. We evaluate on a fixed set of 512 validation prompts per environment and decode
with temperature𝑇 = 0.5 using stochastic sampling. We report success rate as the primary metric
across all environments