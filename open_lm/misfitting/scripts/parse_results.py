import glob
import json
import os
import re
import pandas as pd

MODEL_CONFIGS_FOLDER = "/mmfs1/gscratch/zlab/margsli/gitfiles/open_lm_scaling/open_lm/model_configs"
MODEL_NC = {}
for f in glob.glob(f"{MODEL_CONFIGS_FOLDER}/misfitting_*.json"):
    model_name = os.path.basename(f).replace(".json", "")
    with open(f, 'r') as f:
        jf = json.loads(f.read().strip())
        attn_params = 4 * jf["hidden_dim"] * jf["hidden_dim"] * jf["n_layers"]
        ffn_params = 2 * 4 * jf["hidden_dim"] * jf["hidden_dim"] * jf["n_layers"]
        norm_params = (2 * jf["n_layers"] + 1) * jf["hidden_dim"]
        emb_params = 2 * jf["vocab_size"] * jf["hidden_dim"]
        N = attn_params + ffn_params + norm_params + emb_params
        N_no_emb = attn_params + ffn_params + norm_params
        mask_flops = 6 * jf["seq_len"] * jf["hidden_dim"]
        MODEL_NC[model_name] = {
            "N": N,
            "N_no_emb": N_no_emb,
            "mask_flops": mask_flops,
            "seq_len": jf['seq_len'],
            "n_layers": jf['n_layers'],
            "hidden_dim": jf['hidden_dim'],
            "vocab_size": jf['vocab_size'],
        }
  

LOG_DIR = "/mmfs1/gscratch/zlab/margsli/gitfiles/open_lm_scaling/ds/models"
RESULTS_DIR = LOG_DIR
results_list = []
final_results_list = []
existing_models=[]

sweeps = [f.name for f in os.scandir(LOG_DIR) if f.is_dir()]
print([s for s in sweeps if "MODEL=misfitting12m" in s])
models = {m:s for s in sweeps for m in [f.name for f in os.scandir(os.path.join(LOG_DIR, s)) if f.is_dir()]}

for m, sweep in models.items():
    if m in ["slurm_logs", "open_lm"]:
        continue
    elif "MODEL=misfitting" not in m:
        print(f"Ignoring {sweep}/{m}, pattern not matched")
        continue
    try:
        model_size = re.search('MODEL=misfitting(\w*)_N', m).group(1)
        steps = int(re.search('_NUMSTEPS=(\w*)_L', m).group(1))
        lr = float(re.search('_LR=(\d*\.\d+)$|_LR=(\d*\.\d+)_', m).group(1)
                   or re.search('_LR=(\d*\.\d+)$|_LR=(\d*\.\d+)_EPOCH=10', m).group(2))
    except:
        print(f"{m} does not have required info or is an old run with EPOCH=1")
    results_path = os.path.join(
        LOG_DIR, sweep, m, sweep, m, "checkpoints/results.jsonl"
    )
    params_path = os.path.join(
        LOG_DIR, sweep, m, sweep, m, "params.txt"
    )
    log_path = os.path.join(
        LOG_DIR, sweep, m, sweep, m, "out.log"
    )

    if not os.path.isfile(results_path) or not os.path.isfile(params_path):
        print(f"No logs found for {sweep}/{m}")
        continue

    model_results = []
    tokens = steps * 512 * 2048
    params_dict = {}
    world_size, workers_per_gpu = 4, 2
    with open(params_path, 'r') as f:
        for l in f:
            if l.startswith("workers: "):
                workers_per_gpu = int(l.replace("workers: ", ""))
            if l.startswith("world_size: "):
                world_size = int(l.replace("world_size: ", ""))

    true_epoch_steps = 16 * world_size * workers_per_gpu
    planned_num_epochs = (steps - 1) // true_epoch_steps + 1

    num_epochs = -1
    with open(log_path, 'r') as f:
        for l in f:
            if "Number of checkpoints to be made: " in l:
                num_epochs = int(re.search("Number of checkpoints to be made: (\d+)\.", l).group(1))

    if num_epochs != planned_num_epochs and num_epochs > 2:
        print(f"ERROR: {str(num_epochs)} made, {str(planned_num_epochs)} planned for {sweep}/{m}")
        continue

    steps_at_epoch_end = [true_epoch_steps * (i + 1) for i in range(num_epochs)]
    steps_at_epoch_end[-1] = steps 
 
    with open(results_path, 'r') as f:
        i = 0
        for l in f:
            eval_info = json.loads(l.strip())[0]
            model_info = MODEL_NC[eval_info["model"]]
            if len(steps_at_epoch_end) < i+1:
                print(steps_at_epoch_end, i)
                print(results_path)
                continue
            eval_info.update({
                "id": f"{sweep}/{m}",
                "total_steps": steps, 
                "total_epochs": num_epochs,
                "current_epoch": i + 1, 
                "peak_lr": lr, 
                "sweep": sweep, 
                # "current_steps": (eval_info["train_tokens"]/(512*2048))
                "current_steps": steps_at_epoch_end[i],
                "train_tokens": steps_at_epoch_end[i] * 512 * 2048,
                "portion_steps_elapsed": steps_at_epoch_end[i] / steps, 
                "N": model_info["N"],
                "N_no_emb": model_info["N_no_emb"],
                "D": steps_at_epoch_end[i] * 512 * 2048,
                "C": 6 * model_info["N"] * steps_at_epoch_end[i] * 512 * 2048 + model_info["mask_flops"],
                "C_no_emb": 6 * model_info["N_no_emb"] * steps_at_epoch_end[i] * 512 * 2048 + model_info["mask_flops"],
                "C_6ND": 6 * model_info["N"] * steps_at_epoch_end[i] * 512 * 2048,
                "C_6ND_no_emb": 6 * model_info["N_no_emb"] * steps_at_epoch_end[i] * 512 * 2048,
                "C4 Eval Loss": eval_info["loss"],
            })
            if "train_tokens_fixed" in eval_info:
                if (eval_info["train_tokens_fixed"] != eval_info["train_tokens"] and 
                    eval_info["train_tokens_fixed"] != eval_info["train_tokens"] + 512 * 2048
                ):
                    print(f'D Disagreement: Found disagreement between train token counts in {sweep}/{m}: {eval_info["train_tokens_fixed"]} vs {eval_info["train_tokens"]}, defaulting to {eval_info["train_tokens"]}')
                del eval_info['train_tokens_fixed']
            model_results.append(eval_info)
            i += 1
        num_logs = len(model_results)
        if num_logs != num_epochs:
            print(f"LOG COUNT MISMATCH: {str(num_logs)} logs found, {str(num_epochs)} epochs should have existed for {sweep}/{m}")
            continue
        model_results[-1]["current_steps"] = steps
        model_results[-1]["train_tokens"] = steps * 512 * 2048
        
        results_list.extend(model_results)
        final_results_list.append(model_results[-1])
        if (eval_info["model"], steps, lr) in existing_models:
            print(f"REDUNDANCY: {eval_info['model']}, {steps}, {lr} already found")
        else:
            existing_models.append((eval_info["model"], steps, lr))

columns=[
    "loss", "C4 Eval Loss", "N", "N_no_emb", "D", "C", "C_no_emb", "C_6ND", "C_6ND_no_emb", "portion_steps_elapsed", "model", "train_tokens", "total_steps", "current_steps", "total_epochs", "current_epoch", "peak_lr", "sweep"
]
pd.DataFrame(results_list).to_csv(os.path.join(LOG_DIR, 'results.csv'), columns=columns, index=False)
pd.DataFrame(final_results_list).to_csv(os.path.join(LOG_DIR, 'final_results.csv'), columns=columns, index=False)