# UltraLLaDA

# Eval
- lonbench:
  export PYTHONPATH=/mnt/shared-storage-user/heguangxin/workspace/eval/opencompass-0.4.2:$PYTHONPATH && \
export TIKTOKEN_CACHE_DIR=/mnt/shared-storage-user/heguangxin/workspace/eval/tmp/tiktoken_cache && \
COMPASS_DATA_CACHE='/mnt/shared-storage-user/heguangxin/workspace/datasets/opencompass/' python3 run.py eval/ruler/64K/80k-ntk562-lr2e-5-cp2/eval_llada_ruler_64k_finetune_80k_600step_firsthalf.py --dump-eval-details -r
