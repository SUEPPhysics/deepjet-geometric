import os, sys
import argparse
import numpy as np
import yaml
import torch
from torch_geometric.data import DataLoader
from torch import nn

from deepjet_geometric.datasets import SUEPV1
from SUEPNet import Net
from Disco import distance_corr
from utils import Plotting

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--name', action='store', type=str, help='Input configuration name. Should match the option --out in suep_train.py')
parser.add_argument('--epoch', action='store', default=-1, type=int, help='Which epoch to load. Leave unspecified for last epoch.')
args = parser.parse_args()
out_dir = args.name+"/"

print("Validating model", out_dir)

# initialize plotter
plot = Plotting(save_dir=out_dir)

# input configuration
files = os.listdir(out_dir)
files = [f for f in files if '.yml' in f]
if len(files) > 1: 
    print("Found multiple configuration .yml files in folder, leave only the correct one in it.")
    sys.exit()
config = yaml.safe_load(open(out_dir+files[0]))
print("Loaded configuration", out_dir+files[0])

# pick epoch
epochs = os.listdir(out_dir)
epochs = [f for f in epochs if 'epoch' in f]
epochs = [int(e.split('epoch-')[-1].split('.pt')[0]) for e in epochs]
epochs.sort()
model_path1 = out_dir + 'model1-epoch-' + str(epochs[args.epoch]) + '.pt'
model_path2 = out_dir + 'model2-epoch-' + str(epochs[args.epoch]) + '.pt'
print("Using model file1", model_path1, model_path2)

# initialize dataset
data_test = SUEPV1(config['dataset']['validation'][0])
test_loader = DataLoader(data_test, batch_size=config['training_pref']['batch_size_validation'],shuffle=True,
                         follow_batch=['x_pf'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

#model = DataParallel(model)

# initilize empty model, optimizer, scheduler like in training script
suep1 = Net(out_dim=config['model_pref']['out_dim'], 
           hidden_dim=config['model_pref']['hidden_dim']).to(device)
optimizer1 = torch.optim.Adam(suep1.parameters(), lr=config['training_pref']['learning_rate'])
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])
suep2 = Net(out_dim=config['model_pref']['out_dim'], 
           hidden_dim=config['model_pref']['hidden_dim']).to(device)
optimizer2 = torch.optim.Adam(suep2.parameters(), lr=config['training_pref']['learning_rate'])
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])

# load model, optimizer, scheduler from desired epoch
suep1.load_state_dict(torch.load(model_path1)['model'])
optimizer1.load_state_dict(torch.load(model_path1)['opt'])
scheduler1.load_state_dict(torch.load(model_path1)['lr'])

suep2.load_state_dict(torch.load(model_path2)['model'])
optimizer2.load_state_dict(torch.load(model_path2)['opt'])
scheduler2.load_state_dict(torch.load(model_path2)['lr'])

@torch.no_grad()
def evaluate():
    suep1.eval()
    suep2.eval()
    
    counter = 0
    results = None
    for data in test_loader:
        counter += 1
        print(str(counter*config['training_pref']['batch_size_validation'])+' / '+str(len(test_loader.dataset)),end='\r')
        data = data.to(device)
        
        with torch.no_grad():
            out1 = suep1(data.x_pf,
                       data.x_pf_batch)
            out2 = suep2(data.x_pf,
                       data.x_pf_batch)
            
            # normalize the outputs
            sigmoid = torch.nn.Sigmoid()
            nn1 = out1[0][:,0]
            nn2 = out2[0][:,0]
            nn1 = sigmoid(nn1)
            nn2 = sigmoid(nn2)

            # store predictions from each classifier and true class per event
            if counter == 1: 
                results = np.array([nn1.cpu().numpy(), 
                                    nn2.cpu().numpy(), 
                                    data.y.cpu().numpy()])
            else:    
                batch_results = np.array([nn1.cpu().numpy(), 
                                    nn2.cpu().numpy(), 
                                    data.y.cpu().numpy()])
                results = np.hstack((results, batch_results))
                            
    return results

   
# evaluate model
results = evaluate()

# make some plots
plot.draw_precision_recall(results[0:2],
                           results[2],
                          'ParticleNet',
                          ['Model 1', 'Model 2'])
plot.draw_disco(results, 
                'ParticleNet', 
                ['Model 1', 'Model 2'])