import os, sys
import argparse
import logging
import yaml
import numpy as np
from numpy.testing import assert_almost_equal as is_equal
import onnx
import onnxruntime
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from deepjet_geometric.datasets import SUEPV1
from SUEPNet import Net
import utils

logging.basicConfig(level=logging.DEBUG)

def to_numpy(t):
    return t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert PyTorch SUEPNet to ONNX")
    parser.add_argument('-n', '--name',
                        type=str,
                        help='Input model dir')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        help='Test batch size',
                        default=1)
    parser.add_argument('-e', '--epoch',
                        type=int,
                        help='Epoch to use (default: -1)',
                        default=-1)
    parser.add_argument('-s', '--suppress',
                        action='store_true',
                        help='Suppress checks')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()

    out_dir = args.name
    config = utils.loadConfigFromDir(out_dir)
    model_path = utils.getModelPath(out_dir, args.epoch)
    export_path = model_path.replace('pt', 'onnx')

    print('Converting {} model to ONNX'.format(model_path))

#     ssd_settings = config['ssd_settings']
#     net_channels = net_config['network_channels']
#     input_dimensions = ssd_settings['input_dimensions']
#     jet_size = ssd_settings['object_size']
#     num_workers = config['evaluation_pref']['workers']
#     dataset = config['dataset']['validation'][0]

#     ssd_settings['n_classes'] += 1

    # base = '{}/{}'.format(config['output']['model'], args.model)
    # source_path = '{}.pth'.format(base)
    # export_path = '{}.onnx'.format(base)

    torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on", device)
    
    # initialize dataset
    data_test = SUEPV1(config['dataset']['validation'][0], obj=config['dataset']['obj'])
    loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True,
                             follow_batch=['x_pf'])
    dummy_input = next(iter(loader))
    dummy_input = (dummy_input.x_pf.to(device), dummy_input.x_pf_batch.to(device))

    # initilize model
    net = Net(out_dim=config['model_pref']['out_dim'], 
               hidden_dim=config['model_pref']['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['training_pref']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            config['training_pref']['step_size'], 
                                            config['training_pref']['gamma'])

    # load model, optimizer, scheduler from desired epoch
    net.load_state_dict(torch.load(model_path)['model'])
    optimizer.load_state_dict(torch.load(model_path)['opt'])
    scheduler.load_state_dict(torch.load(model_path)['lr'])
    net.eval()
    
    print('Export as ONNX model')
    
    torch.onnx.export(net,
                      dummy_input,
                      export_path,
                      verbose=True,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    logger.info('Validating graph')
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)

    logger.info('Matching outputs')
    ort_session = onnxruntime.InferenceSession(export_path)
    # Compute PyTorch output prediction
    torch_out = list(map(to_numpy, list(net(dummy_input))))
    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_out = ort_session.run(None, ort_inputs)
    # Compare ONNX Runtime and PyTorch results
    if not args.suppress:
        logger.info('Performing checks')
        for i, task in enumerate(['Localization',
                                  'Classification',
                                  'Regression']):
            is_equal(torch_out[i], ort_out[i], decimal=3)
            logger.info('{} task: OK'.format(task))

    logger.info("Exported model has been successfully tested with ONNXRuntime")
