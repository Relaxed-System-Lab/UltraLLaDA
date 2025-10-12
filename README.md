# UltraLLaDA
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-UltraLLaDA-yellow)](https://huggingface.co/relaxe-system-lab/UltraLLaDA)

We introduce UltraLLaDA , a scaled variant of LLaDA-8B-Base that extends the context length up to 128K tokens with light-weight post-training, enabling long-context comprehension and generation.

# Eval
```bash
export PYTHONPATH=/path/to/eval/opencompass-0.4.2:$PYTHONPATH && \
export TIKTOKEN_CACHE_DIR=/path/to/eval/tiktoken/tiktoken_cache && \
export COMPASS_DATA_CACHE='/path/to/eval/data/'
```
- lonbench:
  ```bash
   python3 run.py /path/to/eval/opencompass-0.4.2/opencompass/eval/longbench/16K/eval_llada_long.py --dump-eval-details -r
  ```

- NIAH:
  ```bash
  python3 run.py /path/to/eval/opencompass-0.4.2/opencompass/eval/niah/128K/eval_llada_niah.py --dump-eval-details -r
  ```

- ruler:
  ```bash
  python3 run.py /path/to/eval/opencompass-0.4.2/opencompass/eval/ruler/32K/eval_llada_ruler.py --dump-eval-details -r
  ```

