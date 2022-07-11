import os, sys, subprocess
import argparse
import yaml

base_config = {
    'dataset': {
        'obj': 'PFcand',
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
        'disco_var' : 'S1',
        'batch_size_train': 512,
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
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=100
#SBATCH --partition=submit-gpu
#SBATCH --gpus-per-node=1

# environment set up
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0

# define that training
echo $SLURM_JOB_ID
touch logs/run_$SLURM_JOB_ID.sh
echo "python {script} --config {config} --out {outDir}; python {eval_script} --name {outDir} --epoch -1" > logs/run_$SLURM_JOB_ID.sh

# run that training
singularity exec --nv --bind {dataDir} /work/submit/bmaier/sandboxes/geometricdl/ /bin/sh logs/run_$SLURM_JOB_ID.sh

srun hostname
srun ls -hrlt
"""

# script parameters
parser = argparse.ArgumentParser(description='Famous Submitter')
parser.add_argument("-f", "--force" , action="store_true", help="Recreate output directories.")
options = parser.parse_args()

# define configurations
dataDir = '/work/submit/lavezzo/debug/'
scripts = ['suep_double_train.py']
disco_vars = ['S1','ntracks']
objs = ['Pfcand', 'bPfcand']
lambdas = [1, 2, 5]

# loop over configurations
for script in scripts:
    for l in lambdas:
        for obj in objs:
            for disco_var in disco_vars:
                
                # define outDir string
                if script != 'suep_single_train.py':
                    # no need to loop over disco_vars if double disco
                    if disco_var != disco_vars[0]: 
                        continue
                    else:
                        outDir = script.split("_")[1] + "_l" + str(l) + "_" + obj
                else:
                    outDir = script.split("_")[1] + "_l" + str(l) + "_" + obj + "_" + disco_var
                
                print(outDir)

                # create outDir
                if os.path.isdir(outDir): 
                    if not options.force:
                        sys.exit("Directory already exists: " + outDir)
                    else:
                        os.system("rm -rf "+str(outDir))
                        os.system("mkdir "+str(outDir))
                else:
                    os.system("mkdir "+str(outDir))

                # modify the config dictionary and save it to outDir
                config = base_config.copy()
                config['training_pref']['lambda_disco'] = l
                config['training_pref']['disco_var'] = disco_var
                config['training_pref']['max_epochs'] = 50
                config['dataset']['obj'] = obj
                configFile = outDir + '/config.yml'
                with open(configFile, 'w') as f: yaml.dump(config, f)

                # modify the running script and save it to outDir
                runner = base_runner.format(dataDir=dataDir, 
                                            outDir=outDir, 
                                            script=script, 
                                            eval_script=script.replace("train", "eval"), 
                                            config=configFile)
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