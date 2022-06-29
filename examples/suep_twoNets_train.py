import os,sys
import argparse
import yaml
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch import nn
from torch_geometric.nn import DataParallel
import matplotlib
matplotlib.use('Agg')

from deepjet_geometric.datasets import SUEPV1
from Disco import distance_corr
from SUEPNet import Net
from utils import Plotting

parser = argparse.ArgumentParser(description='Two networks DiSco.')
parser.add_argument('--config', action='store', type=str, help='Input configuration.')
parser.add_argument('--out', action='store', type=str, help='Output path.')
parser.add_argument('-f', '--force', action='store_true', help='Overwrites output directory if called.')
args = parser.parse_args()

# output directory for model and plots
out_dir = args.out
if os.path.isdir(out_dir): 
    if args.force:
        print("Deleting " + out_dir)
        os.system("rm -r " + out_dir)
        os.system("mkdir " + out_dir)
    else:
        print("This directory already exists. Hope it's not filled.")
plot = Plotting(save_dir=out_dir)
    
# input configuration
config = yaml.safe_load(open(args.config))
# save it to the output folder
os.system("cp " + args.config + " " + out_dir)

data_train = SUEPV1(config['dataset']['train'][0])
data_test = SUEPV1(config['dataset']['validation'][0])

train_loader = DataLoader(data_train, batch_size=config['training_pref']['batch_size_train'],shuffle=True,
                          follow_batch=['x_pf'])
test_loader = DataLoader(data_test, batch_size=config['training_pref']['batch_size_validation'],shuffle=True,
                         follow_batch=['x_pf'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

suep1 = Net(out_dim=config['model_pref']['out_dim'], 
           hidden_dim=config['model_pref']['hidden_dim']).to(device)
suep2 = Net(out_dim=config['model_pref']['out_dim'], 
           hidden_dim=config['model_pref']['hidden_dim']).to(device)

optimizer1 = torch.optim.Adam(suep1.parameters(), lr=config['training_pref']['learning_rate'])
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])

optimizer2 = torch.optim.Adam(suep2.parameters(), lr=config['training_pref']['learning_rate'])
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])


def train(epoch):
    suep1.train()
    suep2.train()
    counter = 0

    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss_disco = 0
    for data in train_loader:
        counter += 1

        print(str(counter*config['training_pref']['batch_size_train'])+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        out1 = suep1(data.x_pf,
                    data.x_pf_batch)
        out2 = suep2(data.x_pf,
                    data.x_pf_batch)

        # ABCDisco loss start
        loss1 = nn.BCEWithLogitsLoss(reduction='mean')(torch.squeeze(out1[0][:,0]).view(-1),data.y.float())
        loss2 = nn.BCEWithLogitsLoss(reduction='mean')(torch.squeeze(out2[0][:,0]).view(-1),data.y.float())
        
        bkgnn1 = out1[0][:,0]
        bkgnn1 = bkgnn1[(data.y==0)]
        bkgnn2 = out2[0][:,0]
        bkgnn2 = bkgnn2[(data.y==0)]
        
        # debug
        sigmoid = torch.nn.Sigmoid()
        bkgnn1 = sigmoid(bkgnn1)
        bkgnn2 = sigmoid(bkgnn2)
        # debug
        
        loss_disco = config['training_pref']['lambda_disco']*distance_corr(bkgnn1,bkgnn2)
        loss = loss1 + loss2 + loss_disco
        # ABCDisco loss end
        
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss_disco += loss_disco.item()
        
    # normalize by number of batches
    total_loss /= counter
    total_loss1 /= counter
    total_loss2 /= counter
    total_loss_disco /= counter
    
    return total_loss, [total_loss1, total_loss2, total_loss_disco]

@torch.no_grad()
def test():
    suep1.eval()
    suep2.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss_disco = 0
    counter = 0
    for data in test_loader:
        counter += 1
        print(str(counter*config['training_pref']['batch_size_train'])+' / '+str(len(test_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out1 = suep1(data.x_pf,
                       data.x_pf_batch)
            out2 = suep2(data.x_pf,
                       data.x_pf_batch)

            loss1 = nn.BCEWithLogitsLoss()(torch.squeeze(out1[0][:,0]).view(-1),data.y.float())
            loss2 = nn.BCEWithLogitsLoss()(torch.squeeze(out2[0][:,0]).view(-1),data.y.float())

            bkgnn1 = out1[0][:,0]
            bkgnn1 = bkgnn1[(data.y==0)]
            bkgnn2 = out2[0][:,0]
            bkgnn2 = bkgnn2[(data.y==0)]
            
            # debug
            sigmoid = torch.nn.Sigmoid()
            bkgnn1 = sigmoid(bkgnn1)
            bkgnn2 = sigmoid(bkgnn2)
            # debug

            loss_disco = config['training_pref']['lambda_disco']**distance_corr(bkgnn1,bkgnn2) 
            
            loss = loss1+loss2+loss_disco
            
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss_disco += loss_disco.item()
        
    # normalize by number of batches
    total_loss /= counter
    total_loss1 /= counter
    total_loss2 /= counter
    total_loss_disco /= counter
    
    return total_loss, [total_loss1, total_loss2, total_loss_disco]


loss_train, loss_val = None, None
partial_losses_train, partial_losses_val = None, None
for epoch in range(0, config['training_pref']['max_epochs']):
    
    # training
    e_loss_train, e_partial_losses_train = train(epoch)
    
    if epoch == 0:
        loss_train = np.array([e_loss_train])
        partial_losses_train = np.array(e_partial_losses_train)
    else:
        loss_train = np.append(loss_train, e_loss_train)
        partial_losses_train = np.vstack((partial_losses_train, e_partial_losses_train))
    
    scheduler1.step()
    scheduler2.step()
    
    # validation
    e_loss_val, e_partial_losses_val = test()
    if epoch == 0:
        loss_val = np.array([e_loss_val])
        partial_losses_val = np.array(e_partial_losses_val)
    else:
        loss_val = np.append(loss_val, e_loss_val)
        partial_losses_val = np.vstack((partial_losses_val, e_partial_losses_val))
    
    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(epoch, e_loss_train, e_loss_val))

    # save model each epoch
    state_dicts = {'model':suep1.state_dict(),
                   'opt':optimizer1.state_dict(),
                   'lr':scheduler1.state_dict()} 
    torch.save(state_dicts, os.path.join(out_dir, 'model1-epoch-{}.pt'.format(epoch)))
    
    state_dicts = {'model':suep2.state_dict(),
                   'opt':optimizer2.state_dict(),
                   'lr':scheduler2.state_dict()} 
    torch.save(state_dicts, os.path.join(out_dir, 'model2-epoch-{}.pt'.format(epoch)))
    
    # update loss plots
    keys = ['Total Loss: C1 + C2 + DiSco']
    plot.draw_loss([loss_train],
                   [loss_val],
                   'total',
                   keys=keys)
    keys = ['Classifier 1', 'lLassifier 2', 'DiSco']
    plot.draw_loss(partial_losses_train.T,
                   partial_losses_val.T,
                   'partials',
                   keys=keys)