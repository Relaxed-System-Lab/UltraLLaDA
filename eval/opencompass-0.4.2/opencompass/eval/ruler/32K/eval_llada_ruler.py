from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k
    

datasets = []
datasets += ruler_datasets_32k

num_gpus = {
    'llada_8b_base': 1,
    # 'ultra_llada': 1,
}

path_dict = {
    'llada_8b_base': 'path/to/llada_8b_base',
    # 'ultra_llada': 'path/to/ultra_llada',
}

models = [
    ('llada_8b_base-o64_b64_s64', {'scaling_factor': 1}, {'steps': 64, 'block_length': 64, }, 64), 
    # ('ultra_llada-o64_b64_s64', {'scaling_factor': 1}, {'steps': 64, 'block_length': 64, }, 64), 
]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_out_len in models
]

work_dir = './outputs/ruler/32k/'

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

# python run.py eval/eval_llada_ruler.py --dump-eval-details -r
# python run.py eval/eval_llada_ruler.py --dump-eval-details -r --debug
