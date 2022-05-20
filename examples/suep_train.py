import os,sys
import argparse
import yaml
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch import nn

from deepjet_geometric.datasets import SUEPV1
from Disco import distance_corr
from SUEPNet import Net

parser = argparse.ArgumentParser(description='Test.')
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
    else:
        print("This directory already exists. Make a new name.")
        sys.exit()
os.system("mkdir " + out_dir)
    
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

#model = DataParallel(model)

suep = Net().to(device)
#suep.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(suep.parameters(), lr=config['training_pref']['learning_rate'])
#optimizer = torch.optim.Adam(disco.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])

def train():
    suep.train()
    counter = 0

    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss_disco = 0
    for data in train_loader:
        counter += 1

        print(str(counter*config['training_pref']['batch_size_train'])+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        optimizer.zero_grad()
        out = suep(data.x_pf,
                    data.x_pf_batch)

        # ABCDisco loss start
        loss1 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,0]).view(-1),data.y.float())
        loss2 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,1]).view(-1),data.y.float())
        
        bkgnn1 = out[0][:,0]
        bkgnn1 = bkgnn1[(data.y==0)]
        bkgnn2 = out[0][:,1]
        bkgnn2 = bkgnn2[(data.y==0)]
        loss_disco = config['training_pref']['lambda_disco']*distance_corr(bkgnn1,bkgnn2)
        loss = loss1 + loss2 + loss_disco
        # ABCDisco loss end

        loss.backward()
        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss_disco += loss_disco.item()

        optimizer.step()

    total_loss /= len(train_loader.dataset)
    total_loss1 /= len(train_loader.dataset)
    total_loss2 /= len(train_loader.dataset)
    total_loss_disco /= len(train_loader.dataset)
    
    return total_loss, [total_loss1, total_loss2, total_loss_disco]

@torch.no_grad()
def test():
    suep.eval()
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
            out = suep(data.x_pf,
                       data.x_pf_batch)

            loss1 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,0]).view(-1),data.y.float())
            loss2 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,1]).view(-1),data.y.float())

            bkgnn1 = out[0][:,0]
            bkgnn1 = bkgnn1[(data.y==0)]
            bkgnn2 = out[0][:,1]
            bkgnn2 = bkgnn2[(data.y==0)]

            loss_disco = config['training_pref']['lamdba_disco']**distance_corr(bkgnn1,bkgnn2) 
            loss = loss1 + loss2 + loss_disco
            
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss_disco += loss_disco.item()

        
    total_loss /= len(train_loader.dataset)
    total_loss1 /= len(train_loader.dataset)
    total_loss2 /= len(train_loader.dataset)
    total_loss_disco /= len(train_loader.dataset)
    
    return total_loss, [total_loss1, total_loss2, total_loss_disco]


loss_train, loss_val = torch.empty(1, 0), torch.empty(1, 0)
partial_losses_train, partial_losses_val = torch.empty(3, 0), torch.empty(3, 0)
for epoch in range(0, 50):
    
    e_loss_train, e_partial_losses_train = train()
    
    # debug
    print()
    print(epoch, e_loss, e_loss_val)
    print()
    
    loss_train = torch.cat((loss_train, e_loss_train), 1)
    partial_losses_train = torch.cat((partial_losses_train, e_partial_losses_train), 1)
    
    scheduler.step()
    
    e_loss_val, e_partial_losses_val = test()
    loss_val = torch.cat((loss_val, e_loss_val), 1)
    partial_losses_val = torch.cat((partial_losses_val, e_partial_losses_val), 1)

    # debug
    print()
    print(epoch, e_loss, e_loss_val)
    print()
    
    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(epoch, e_loss_train, e_loss_val))

    state_dicts = {'model':suep.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(out_dir, 'epoch-{}.pt'.format(epoch)))
    
    # update loss, metrics plots
    keys = ['Total Loss: C1 + C2 + DiSco']
    plot.draw_loss(train_loss.cpu().numpy(),
                   val_loss.cpu().numpy(),
                   name,
                   keys=keys)
    
    keys = ['Classifier 1', 'CLassifier 2', 'DiSco']
    plot.draw_loss(train_loss.cpu().numpy(),
                   val_loss.cpu().numpy(),
                   name,
                   keys=keys)