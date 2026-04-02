# ray stop --force || true

python main.py \
  +experiments=ReAct/ablations/no-history \
  ++llm.llm_model=/projects/e32695/huggingface/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  ++llm.use_vllm=true \
  ++llm.feedback_steps=1 \
  ++llm.vllm_enforce_eager=true \
  ++llm.vllm_gpu_memory_utilization=0.3 \
  ++llm.vllm_enable_prefix_caching=false \
  ++llm.vllm_max_model_len=8192 \
  ++llm.vllm_max_num_batched_tokens=8192 \
  ++llm.vllm_max_num_seqs=16 \
  ++game.max_steps=5 \
  ++game.max_step_multiplier=1 \
  ++evaluation.environment_names=[synchronous/0_cheese_sandwich] \
  ++evaluation.testing_seeds=[0] \
  ++llm.max_feedback_steps=10 \
  ++llm.feedback_attempts=10 \
  ++game.record=false \
