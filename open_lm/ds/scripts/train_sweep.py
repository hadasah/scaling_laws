import argparse
import numpy as np
import os
from datetime import datetime
from copy import copy
from open_lm.constants.slurm_constants import PROJECT_SPECS_DICT
from open_lm.ds.scripts.slurm_job import run_grid

PROJECT = "ds"
DEFAULT_DIR_PATH ='/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-4])
SWEEP_NAME_DEFAULT = 'test'
DEFAULT_JOBTIME = '24:00:00'

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    add_time_to_name='front',
    project=PROJECT,
    account=None, 
    partition=None,
):

    name_keys = ["MODEL", "NUM_STEPS", "LR"]
    PROJECT_SPECS = PROJECT_SPECS_DICT[project]
    t = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    SWEEP_NAME = sweep_name
    if add_time_to_name == 'front':
        SWEEP_NAME = f"{t}_sweep_{SWEEP_NAME}"


    for model in PROJECT_SPECS:
        if model == "default":
            continue
        
        SPECS = copy(PROJECT_SPECS["default"])
        SPECS.update(PROJECT_SPECS[model])
        NUM_GPUS = SPECS['NUM_GPUS']
        NUM_NODES = (SPECS['NUM_GPUS'] - 1) // 8 + 1
        MODEL_DIR = f"{SPECS['MODEL_DIR']}/{SWEEP_NAME}"

        grids = {
            SWEEP_NAME: {
                'positional_args': {
                    "SWEEP_NAME": [SWEEP_NAME],
                    "NUM_GPUS": [SPECS['NUM_GPUS']],
                    "WANDB_PROJECT": [SPECS['WANDB_PROJECT']],
                    "HF_MODEL": ["None"],
                    "MODEL": [model],
                    "TRAIN_DATA": ["None"],
                    "TRAIN_DATA_MANIFEST": [SPECS["TRAIN_DATA_MANIFEST"]],
                    "VAL_DATA": [SPECS['VAL_DATA']],
                    "MODEL_DIR": [MODEL_DIR],
                    "SEQ_LEN": [SPECS['SEQ_LEN']],
                    "NUM_WARMUP_STEPS": [SPECS['NUM_WARMUP_STEPS']],
                    "GLOBAL_BATCH_SIZE": [SPECS['GLOBAL_BATCH_SIZE']],
                    "GLOBAL_VAL_BATCH_SIZE": [SPECS['GLOBAL_VAL_BATCH_SIZE']],
                    "NUM_STEPS": SPECS['NUM_STEPS'],
                    "ACCUM_FREQ": [SPECS['ACCUM_FREQ']],
                    "USE_FSDP": [SPECS["USE_FSDP"]],
                    "LOG_INTERVAL": [SPECS['LOG_INTERVAL']],
                    "VALIDATION_INTERVAL": [SPECS['VALIDATION_INTERVAL']],
                    "LR": SPECS['LR'],
                    "END_LR_RATIO": [SPECS['END_LR_RATIO']],
                    "CLIP_NORM": [SPECS['CLIP_NORM']],
                    "WD": [SPECS['WD']],
                    "ADAM_BETA2": [SPECS['ADAM_BETA2']],
                    "EPOCH_STEPS": SPECS['EPOCH_STEPS'],
                    "PROJECT_DIR": [SPECS["PROJECT_DIR"]],
                },
            },
        }

        for sweep_name, grid in grids.items():
            run_grid(
                grid,
                name_keys,
                sweep_name,
                user=os.environ['USER'],
                prefix=f'bash {SPECS.get('PROJECT_DIR')}/open_lm/ds/scripts/train.sh',
                gpus=NUM_GPUS,
                cpus=SPECS["NUM_CPUS"],
                nodes=NUM_NODES,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=SPECS.get("JOBTIME", DEFAULT_JOBTIME),            
                saveroot=MODEL_DIR,
                logroot=MODEL_DIR,
                mem_gb=SPECS.get("MEM_GB"),
                requeue=True,
                add_name='end',
                repo_name="open_lm",
                conda_env_name=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--project', type=str, default=PROJECT)
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)

    args = parser.parse_args()
    main(
        sweep_name=args.sweep_name,
        add_time_to_name=args.add_time_to_name,
        project=args.project,
        account=args.slurm_account, 
        partition=args.slurm_partition,
    )
