"""Contains basic helper functions for running a parameter sweep on the Hyak
cluster using the SLURM scheduler.
Adapted from ParlAI
"""

from collections import namedtuple
import json
import os
import subprocess
import sys
import random
import hashlib

from open_lm.constants.slurm_constants import CONSTANTS

username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")

DEFAULT_DIR_PATH ='/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-4])
BASH_IF_CLAUSE = """
if [[ "$SLURM_ARRAY_TASK_ID" == "{index}" ]]; then
    srun -K1 bash {SAVE}/run.sh > {SAVE}/stdout.$SLURM_ARRAY_TASK_ID 2> {SAVE}/stderr.$SLURM_ARRAY_TASK_ID
fi
"""
SLRM_JOB_ARRAY_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={SWEEP_NAME}
#SBATCH --output={SAVE_ROOT}/slurm_logs/stdout.%j
#SBATCH --error={SAVE_ROOT}/slurm_logs/stderr.%j
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --open-mode=append
#SBATCH --nodes={nodes}
#SBATCH --time={jobtime}
#SBATCH --signal=USR1@60
#SBATCH --cpus-per-task={cpus}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={mem_gb}G
{SBATCH_EXTRAS}

source ~/.bashrc
{conda_command}

echo "# -------- BEGIN CALL TO run.sh --------"
{JOB_LAUNCHER}
"""

SH_TEMPLATE = """
#!/bin/bash
set -e

# stores the child process
CHILD=""

# handles a TERM signal
term_handler () {{
    wait "$CHILD"
}}

# handles an interrupt (aka ctrl-C)
int_handler () {{
    kill -s INT "$CHILD"
    wait "$CHILD"
}}

usr1_handler () {{
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)."
    kill -s INT "$CHILD"  # send ctrl-c to python
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then
        echo "Waiting 5s and resubmitting..."
        sleep 5
        echo "Resubmitting..."
        scontrol requeue $SLURM_JOB_ID
    fi
    wait "$CHILD"
}}

trap 'int_handler' INT
trap 'usr1_handler' USR1
trap 'term_handler' TERM

export NCCL_SOCKET_IFNAME=^docker0,lo

cd {NEW_DIR_PATH}
export PYTHONPATH={SAVE_ROOT}/{repo_name}:$PYTHONPATH
{python_cmd} 

CHILD="$!"
wait "$CHILD"
sleep 30
"""

def run_grid(
    grid,
    name_keys,
    sweep_name,
    user=os.environ['USER'],
    prefix=None,
    gpus=1,
    cpus=10,
    nodes=1,
    account=None,
    partition=None,
    DIR_PATH=DEFAULT_DIR_PATH,
    jobtime='01:59:59',
    saveroot=None,
    logroot=None,
    mem_gb=64,
    requeue=False,
    add_name=None,
    repo_name="code",
    conda_env_name=None,
):
    """Generates full commands from a grid.

    Arguments:
    grid -- (dict) keys are hyperparam strings (e.g. --learningrate or -lr),
        values are lists of parameter options (e.g. [0.5, 0.05, 0.005]).
        You can tie options together in a limited fashion (e.g.
        '--opt': ['sgd -lr 0.5', 'adam -lr 0.005']), but we don't support
        nesting dicts/lists yet.
    name_keys -- (set) contains any params to always include in the model
        filename (e.g. {'-hs'} will make sure that the filename includes
        _hs=X_). By default, any key with more than one value will also be
        included in the model filename.
    sweep_name -- (str) name of the sweep
    user -- (str) user name to use for save directory (default $USER)
    prefix -- (str) base command to run
    dataparallel -- (bool) set to True if running with nn.DataParallel
    add_name -- (str) "end" or None, indicating whether to
        add the name to the command and if so, where
    """
    if not prefix:
        raise ValueError('Need prefix command')
    SAVE_ROOT = saveroot if saveroot is not None else save_root(sweep_name, user)
    LOG_ROOT = logroot if logroot is not None else log_root(sweep_name, user)

    Job = namedtuple('Job', ['cmd', 'name'])
    all_jobs = [Job(cmd=prefix, name='')]

    for key, args in grid.get('positional_args', {}).items():
        new_jobs = []
        # save_name
        save_key = key
        while save_key.startswith('-'):
            save_key = save_key[1:]
        save_key = save_key.replace('_', '')

        for job in all_jobs:
            for a in args:
                new_cmd = ' '.join((job.cmd, str(a)))
                new_name = job.name
                if (len(args) > 1 or key in name_keys):
                    if a is None:
                        new_jobs.append(Job(cmd=new_cmd, name=new_name))
                        continue
                    if type(a) == str:
                        a = a.replace('_', '')
                        if ' ' in a:
                            a = a.replace(' --', '_').replace(' -', '_')
                            a = a.replace(' ', '=')
                    new_name += '_{}={}'.format(save_key, a)
                new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    for key, args in grid.get('named_args', {}).items():
        new_jobs = []
        # save_name
        save_key = key
        while save_key.startswith('-'):
            save_key = save_key[1:]
        save_key = save_key.replace('_', '')

        for job in all_jobs:
            for a in args:
                new_cmd = ' '.join((job.cmd, str(key), str(a)))
                new_name = job.name
                if (len(args) > 1 or key in name_keys):
                    if len(a) == 0:
                        new_jobs.append(Job(cmd=new_cmd, name=new_name))
                        continue
                    if type(a) == str:
                        a = a.replace('_', '')
                        if ' ' in a:
                            a = a.replace(' --', '_').replace(' -', '_')
                            a = a.replace(' ', '=')
                    new_name += '_{}={}'.format(save_key, a)
                new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    final_jobs = []
    job_id = 1
    for job in all_jobs:
        new_cmd = job.cmd
        new_name = job.name[1:]
        if add_name:
            new_cmd = ' '.join((new_cmd, new_name))
        final_jobs.append(Job(cmd=new_cmd, name=new_name))
        job_id += 1

    NEW_DIR_PATH = DIR_PATH

    # Dump grid to grid file
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    with open(os.path.join(SAVE_ROOT, 'grid.json'), 'w') as f:
        json.dump(grid, f)

    jobs_path = []
    for job in final_jobs:
        jobs_path.append(
            create_job_files(
                sweep_name,
                SAVE_ROOT,
                LOG_ROOT,
                job.name,
                job.cmd,
                gpus=gpus,
                nodes=nodes,
                requeue=requeue,
                NEW_DIR_PATH=NEW_DIR_PATH,
                repo_name=repo_name,
            )
        )
    submit_array_jobs(
        SWEEP_NAME=sweep_name,
        SAVE_ROOT=SAVE_ROOT,
        gpus=gpus,
        cpus=cpus,
        nodes=nodes,
        account=account,
        partition=partition,
        jobtime=jobtime,
        DIR_PATH=DIR_PATH,
        mem_gb=mem_gb,
        requeue=requeue,
        NEW_DIR_PATH=NEW_DIR_PATH,
        jobs_path=jobs_path,
        conda_env_name=conda_env_name,
    )


def bash(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip('b').strip('\'').strip('"')
    return output


def save_root(SWEEP_NAME, unixname):
    """Return root folder for saving model files, stdout, stderr, etc."""
    SAVE_ROOT = os.path.join(RUN_CONSTANTS.get('MODEL_FOLDER'), SWEEP_NAME)
    return SAVE_ROOT

def log_root(SWEEP_NAME, unixname):
    """Return root folder for saving tensorboard logs"""
    LOG_ROOT = os.path.join(RUN_CONSTANTS.get('LOG_FOLDER'), SWEEP_NAME)
    return LOG_ROOT


def create_job_files(
    SWEEP_NAME,
    SAVE_ROOT,
    LOG_ROOT,
    job_name,
    python_cmd,
    gpus=1,
    nodes=1,
    requeue=False,
    NEW_DIR_PATH=DEFAULT_DIR_PATH,
    repo_name="",
):
    """Creates job folders and scripts"""
    SHOULD_REQUEUE = str(requeue).lower()
    SAVE = os.path.join(SAVE_ROOT, job_name)
    bash('mkdir -p ' + SAVE)
    LOG = os.path.join(LOG_ROOT, job_name)
    bash('mkdir -p ' + LOG)
    SCRIPTFILE = os.path.join(SAVE, 'run.sh')
    ARRAYJOBFILE = os.path.join(SAVE_ROOT, 'array_jobs')

    if not gpus:
        ntasks_per_node = 1
    else:
        if gpus > 8:
            ntasks_per_node = 8
        else:
            ntasks_per_node = gpus
    with open(SCRIPTFILE, 'w') as fw:
        fw.write(SH_TEMPLATE.format(**locals()).lstrip())
    return SAVE


def submit_array_jobs(
    SWEEP_NAME,
    SAVE_ROOT,
    gpus=1,
    cpus=1,
    nodes=1,
    account=None,
    partition=None,
    jobtime=None,
    DIR_PATH=DEFAULT_DIR_PATH,
    mem_gb=64,
    requeue=False,
    NEW_DIR_PATH=DEFAULT_DIR_PATH,
    jobs_path=[],
    conda_env_name=None,
    append_to_sbatch_str=None,
):
    SLURMFILE = os.path.join(SAVE_ROOT, 'run.slrm')
    if not gpus:
        ntasks_per_node = 1
    else:
        if gpus > 8:
            ntasks_per_node = 8
        else:
            ntasks_per_node = gpus
    SBATCH_EXTRAS = []
    total_num_jobs = len(jobs_path) - 1
    # Request the number of GPUs (defaults to 1)
    if gpus > 0:
        if gpus > 8:
            gpustr = '#SBATCH --gpus-per-node=8'
        else:
            gpustr = '#SBATCH --gpus-per-node={}'.format(gpus)
        SBATCH_EXTRAS.append(gpustr)

    conda_command = f'conda activate {conda_env_name}' if conda_env_name else ''
    # make sure sbatch extras are a string
    SBATCH_EXTRAS = "\n".join(SBATCH_EXTRAS)
    JOB_LAUNCHER = []
    for idx, each_path in enumerate(jobs_path):
        JOB_LAUNCHER.append(BASH_IF_CLAUSE.format(index=idx, SAVE=each_path, nodes=nodes))
    JOB_LAUNCHER = "\n".join(JOB_LAUNCHER)
    bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'slurm_logs'))
    with open(SLURMFILE, 'w') as fw:
        fw.write(SLRM_JOB_ARRAY_TEMPLATE.format(**locals()).lstrip())