import os, sys, subprocess
import yaml

base_config = {
    'dataset': {
        'train': ['/work/submit/lavezzo/debug/'], 
        'validation': ['/work/submit/lavezzo/debug/val/'], 
        'test': ['/work/submit/lavezzo/debug/val/']
    }, 
    'model_pref': {
        'hidden_dim': 16,
        'out_dim': 1
    }, 
    'training_pref': {
        'lambda_disco': 1,
        'dico_var' : 'ntracks',
        'batch_size_train': 1024,
        'batch_size_validation': 1024,
        'max_epochs': 50,
        'learning_rate': 0.001,
        'step_size': 20,
        'gamma': 0.5},
    'evaluation_pref': {
        'batch_size': 50,
        'names_classes': ['SUEP', 'QCD'],
        'workers': 0}
}

base_runner = """#!/bin/bash
#
#SBATCH --job-name=SUEPNet_train
#SBATCH --output=logs/res_%j.txt
#SBATCH --error=logs/err_%j.txt
#
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=100
#SBATCH --partition=submit-gpu
#SBATCH --gpus-per-node=1

# environment set up
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0

# define that training
touch logs/run_$(SLURM_JOB_ID).sh
echo "python {script} --config {config} --out {outDir}" > logs/run_%j.sh

# run that training
singularity exec --nv --bind {dataDir} /work/submit/bmaier/sandboxes/geometricdl/ /bin/sh logs/run_%j.sh

srun hostname
srun ls -hrlt
"""

# define configurations
dataDir = '/work/submit/lavezzo/debug/'
script = 'suep_single_train.py'
lambdas = [1, 2, 3]

# loop over configurations
for l in lambdas:
    
    # define outDir string
    outDir = 'single_S1_l'+str(l)
    if os.path.isdir(outDir): 
        sys.exit("Directory already exists: " + outDir)
    else:
        os.system("mkdir "+str(outDir))
    
    # modify the config dictionary and save it to outDir
    config = base_config.copy()
    config['training_pref']['lambda_disco'] = l
    configFile = outDir + '/config.yaml'
    with open(configFile, 'w') as f: yaml.dump(config, f)
    
    # modify the running script and save it to outDir
    runner = base_runner.format(dataDir=dataDir, outDir=outDir, script=script, config=configFile)
    runnerFile = outDir + '/train.sh'
    with open(runnerFile, 'w') as f: f.write(runner)
    
    # for each, submit a training job
    htc = subprocess.Popen(
        "sbatch " + runnerFile,
        shell  = True,
        stdin  = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        close_fds=True
    )
    out, err = htc.communicate()
    print(out.decode('utf-8'))