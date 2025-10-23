from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM
from opencompass.models import LLaDACausalLMBatch

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

# -------------------------
# Model registry
# -------------------------
# How many GPUs to use per model
num_gpus = {
    'llada_8b_base': 1,
    # 'ultra_llada': 1,
}

path_dict = {
    'llada_8b_base': 'path/to/llada_8b_base',
    # 'ultra_llada': 'path/to/ultra_llada',
}

models = [
    ('llada_8b_base-o32_b32_s16', {'scaling_factor': 1}, {'steps': 16, 'block_length': 32, }, 32), 
    # ('ultra_llada-o32_b32_s16', {'scaling_factor': 1}, {'steps': 16, 'block_length': 32, }, 32), 
]

# Instantiate OpenCompass model entries
models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_out_len in models
]

work_dir = './outputs/niah/128k/'


infer = dict(
    partitioner=dict(type=NaivePartitioner), 
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
