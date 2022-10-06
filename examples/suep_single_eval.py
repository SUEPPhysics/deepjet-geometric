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
import utils
from utils import Plotting

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--name', action='store', required=True, type=str, help='Input configuration name. Should match the option --out in suep_train.py')
parser.add_argument('--epoch', action='store', default=-1, type=int, help='Which epoch to load. Leave unspecified for last epoch.')
args = parser.parse_args()
out_dir = args.name+"/"

print("Validating model", out_dir)

# initialize plotter
plot = Plotting(save_dir=out_dir)

# input configuration
config = utils.loadConfigFromDir(out_dir)

# pick epoch
model_path = utils.getModelPath(out_dir, args.epoch)

# initialize dataset
data_test = SUEPV1(config['dataset']['test'][0], obj=config['dataset']['obj'])
test_loader = DataLoader(data_test, batch_size=config['training_pref']['batch_size_validation'],shuffle=True,
                         follow_batch=['x_pf'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

#model = DataParallel(model)

# initilize empty model, optimizer, scheduler like in training script
suep = Net(out_dim=config['model_pref']['out_dim'], 
           hidden_dim=config['model_pref']['hidden_dim']).to(device)
optimizer = torch.optim.Adam(suep.parameters(), lr=config['training_pref']['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])

# load model, optimizer, scheduler from desired epoch
suep.load_state_dict(torch.load(model_path)['model'])
optimizer.load_state_dict(torch.load(model_path)['opt'])
scheduler.load_state_dict(torch.load(model_path)['lr'])

@torch.no_grad()
def evaluate():
    suep.eval()
    
    counter = 0
    results = None
    for data in test_loader:
        counter += 1
        print(str(counter*config['training_pref']['batch_size_validation'])+' / '+str(len(test_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = suep(data.x_pf,
                       data.x_pf_batch)
            
            # normalize the outputs
            sigmoid = torch.nn.Sigmoid()
            nn1 = out[0][:,0]
            nn1 = sigmoid(nn1)
            
            # count number of tracks in each event
            if config['training_pref']['disco_var'] == 'ntracks':
                ntracks = torch.cuda.FloatTensor([np.count_nonzero(data.x.cpu().numpy()[x,:,0]) for x in range(data.x.shape[0])])
                disco_var = ntracks
            elif config['training_pref']['disco_var'] == 'S1':
                S1 = data.S1
                disco_var = S1

            # store predictions from each classifier and true class per event
            if counter == 1: 
                results = np.array([nn1.cpu().numpy(), 
                                    disco_var.cpu().numpy(),
                                    data.y.cpu().numpy()])
            else:    
                batch_results = np.array([nn1.cpu().numpy(), 
                                    disco_var.cpu().numpy(),
                                    data.y.cpu().numpy()])
                results = np.hstack((results, batch_results))
                                
    return results

   
# evaluate model
results = evaluate()

# make some plots
plot.draw_precision_recall(results[0],
                           results[2],
                          'ParticleNet',
                          ['Model 1'])
plot.draw_disco(results, 
                'ParticleNet', 
                ['Model 1', '# Tracks'],
               xlim=[0,1],
               ylim=[0,500])