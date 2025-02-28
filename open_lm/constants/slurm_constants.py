CONSTANTS = {
    None: { ## TODO: FILL IN
        "MODEL_FOLDER": None, ## TODO: FILL IN
        "LOG_FOLDER": None, ## TODO: FILL IN
    },
}

PROJECT_SPECS_DICT = {
    "ds":{
        "default": {
            "WANDB_PROJECT": None, ## TODO: FILL IN
            "CONDA_ENV": None, ## TODO: FILL IN
            "PROJECT_DIR": None, ## TODO: FILL IN
            "NUM_CPUS": 4,
            "MEM_GB": 128,
            "JOBTIME": '12:00:00',
            "SLURM_ACCOUNT": None, ## TODO: FILL IN
            "SLURM_PARTITION": None, ## TODO: FILL IN
            "NUM_GPUS": 4,
            "WANDB_PROJECT": None, ## TODO: FILL IN
            "MODEL": ['misfitting_{}m'.format(str(num)) for num in [12, 17, 25, 35, 50, 70, 100, 150, 200, 300, 400]],
            "TRAIN_DATA": None, ## TODO: FILL IN
            "TRAIN_DATA_MANIFEST": None, ## TODO: FILL IN
            "VAL_DATA": None, ## TODO: FILL IN
            "MODEL_DIR": None, ## TODO: FILL IN,
            "SEQ_LEN": 2048,
            "NUM_WARMUP_STEPS": 50,
            "GLOBAL_BATCH_SIZE": 512,
            "GLOBAL_VAL_BATCH_SIZE": "None",
            "ACCUM_FREQ": 8,
            "USE_FSDP": "False",
            "LOG_INTERVAL": 20,
            "VALIDATION_INTERVAL": 1000,
            "END_LR_RATIO": 0.1,
            "CLIP_NORM": 1,
            "WD": 0.1,
            "ADAM_BETA2": 0.95,
            "EPOCH_STEPS": [64],
        },
        "misfitting_12m": {
            "NUM_STEPS": [100, 200, 250, 360, 500, 750, 1000, 4000, 6122, 7346, 8888, 11851],
            "LR": [0.002, 0.004, 0.008],
            "ACCUM_FREQ": 32,
        },
        "misfitting_17m": {
            "NUM_STEPS": [100, 200, 250, 500, 750, 1000, 1250, 1500, 10581],
            "LR": [0.001, 0.002, 0.004, 0.008],
            "ACCUM_FREQ": 32,
        },
        "misfitting_25m": {
            "NUM_STEPS": [250, 360, 500, 750, 1000, 1500, 2000, 8000, 16000],
            "LR": [0.001, 0.002, 0.0004],
            "ACCUM_FREQ": 32,
        },
        "misfitting_35m": {
            "NUM_STEPS": [200, 250, 360, 500, 750, 1000, 1250, 1500, 4000, 16000],
            "LR": [0.001, 0.002, 0.004],
            "ACCUM_FREQ": 16,
        },
        "misfitting_50m": { 
            "NUM_STEPS": [250, 500, 750, 1000, 1250, 1500, 1800, 2000, 2500, 4000, 16000],
            "LR": [0.001, 0.002, 0.004],
            "ACCUM_FREQ": 16,
        },
        "misfitting_70m": { 
            "NUM_STEPS": [500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 5000],
            "LR": [0.001, 0.002, 0.004],
            "ACCUM_FREQ": 32,
        },
        "misfitting_100m": {
            "NUM_STEPS": [250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 12000],
            "LR": [0.001, 0.002, 0.004],
            "ACCUM_FREQ": 32,
        },
        "misfitting_150m": {
            "NUM_STEPS": [250, 500, 1000, 3000, 4000, 8000, 12000],
            "LR": [0.001, 0.002, 0.004],
            "ACCUM_FREQ": 32,
        },
        "misfitting_200m": {
            "NUM_STEPS": [500, 1500, 3000, 4000, 6000, 9000],
            "LR": [0.0004, 0.001, 0.002],
            "ACCUM_FREQ": 8,
        },
        "misfitting_300m": { 
            "NUM_STEPS": [4000, 6000, 8000],
            "LR": [0.0004, 0.001, 0.002],
            "ACCUM_FREQ": 32,
        },
        "misfitting_400m": {
            "NUM_STEPS": [3000, 6000, 8000, 12000],
            "LR": [0.002, 0.0004],
            "NUM_GPUS": 8,
            "ACCUM_FREQ": 8,
        },
        "misfitting_1b": {
            "NUM_STEPS": [10000, 20000, 40000],
            "LR": [0.0004, 0.001],
            "NUM_GPUS": 16,
            "ACCUM_FREQ": 2,
            "GLOBAL_VAL_BATCH_SIZE": 16,
            "NUM_CPUS": 6,
            "MEM_GB": 512,
            "USE_FSDP": "True",
        },
        "misfitting_2b": {
            "NUM_STEPS": [40000],
            "LR": [0.0004, 0.001],
            "NUM_GPUS": 16,
            "ACCUM_FREQ": 4,
            "GLOBAL_VAL_BATCH_SIZE": 16,
            "NUM_CPUS": 6,
            "MEM_GB": 900,
            "USE_FSDP": "True",
        },
    }, 

}