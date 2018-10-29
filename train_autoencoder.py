from datetime import datetime
from functools import partial
from argparse import ArgumentParser
from extreme_classification.loaders import LibSVMLoader
from extreme_classification.autoencoders import Autoencoder as AE
from extreme_classification.clusterings import CoOccurrenceAgglomerativeClustering as CAAC

import sys
import yaml
import torch
import numpy as np


TIME_STAMP = datetime.utcnow().isoformat()


def train_model(model, optimizer, data_loader, epochs, interval, cur_device, perm_func=None):
    for epoch in range(0, epochs):
        model.train()
        for batch_idx, (inp, tgt) in enumerate(data_loader):
            if perm_func is None:
                inp = inp.squeeze(1)
                data = inp.to(device=cur_device, dtype=torch.float32)
            else:
                tgt = tgt.squeeze(1)
                data = perm_func(tgt).to(device=cur_device, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, data)
            loss.backward()
            optimizer.step()
            if batch_idx % interval == 0:
                print('Training epoch {} : Loss = {}'.format(epoch, loss.item()))
    return model


def get_data_loader(data_root, cuda_opt, batch_size, for_outputs):
    loader_kwargs = {}
    if cuda_opt:
        loader_kwargs = {'num_workers': 1, 'pin_memory': True}

    loader = LibSVMLoader(data_root)
    data_loader = torch.utils.data.DataLoader(loader, batch_size=batch_size, shuffle=True,
                                              **loader_kwargs)

    get_permutation = None
    if for_outputs:
        agglo_clusters = CAAC(loader.get_classes())
        ordering = agglo_clusters.get_ordering()
        get_permutation = partial(np.take, indices=ordering, axis=1)

    return data_loader, get_permutation


def save_model(model, for_outputs):
    # Transfer the model to CPU before saving
    split_str = 'outputs' if for_outputs else 'inputs'
    torch.save(model.to('cpu'),
               'trained_model_{}_{}.pt'.format(split_str, TIME_STAMP))


def get_repr(model, data_loader, cur_device, perm_func=None):
    reprs = []
    model.eval()
    for (inp, tgt) in data_loader:
        if perm_func is None:
            inp = inp.squeeze(1)
            reprs.append(model(inp.to(device=cur_device, dtype=torch.float32)))
        else:
            tgt = tgt.squeeze(1)
            reprs.append(model(perm_func(tgt).to(device=cur_device, dtype=torch.float32)))
    reprs = torch.cat(reprs)
    return reprs


if __name__ == '__main__':
    parser = ArgumentParser()

    # data argument
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root folder for dataset')
    parser.add_argument('--for_outputs', action='store_true',
                        help='If toggled, then the autoencoder is trained on the outputs, \
                              otherwise it is trained on the outputs')

    # architecture arguments
    parser.add_argument('--encoder_cfg', type=str, required=True,
                        help='Encoder architecture configuration in YAML format')
    parser.add_argument('--decoder_cfg', type=str, required=True,
                        help='Decoder architecture configuration in YAML format')

    # training configuration arguments
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device string <device_name>:<device_id>')
    parser.add_argument('--seed', type=int, default=None,
                        help='Manually set the seed for the experiments for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the autoencoder for')
    parser.add_argument('--interval', type=int, default=-1,
                        help='Interval between two status updates on training')

    # optimizer arguments
    parser.add_argument('--optimizer_cfg', type=str, required=True,
                        help='Optimizer configuration in YAML format')

    # post training options
    parser.add_argument('--save_model', action='store_true',
                        help='Toggle to save model after training')
    parser.add_argument('--get_repr', action='store_true',
                        help='Toggle to save representation of dataset')
    args = parser.parse_args()

    # Check CUDA availability
    USE_CUDA = args.device.find('cuda')
    if not torch.cuda.is_available() and USE_CUDA:
        print("Attempting to use CUDA when CUDA is unavailable!!")
        sys.exit(1)

    cur_device = torch.device(args.device)

    # Seeding for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if USE_CUDA:
            torch.cuda.manual_seed(args.seed)

    # Get the models and the optimizers required
    model = AE(yaml.load(open(args.encoder_cfg)), yaml.load(open(args.decoder_cfg))).to(cur_device)

    opt_config = yaml.load(open(args.optimizer_cfg))
    optimizer = getattr(torch.optim, opt_config['name'])(model.parameters(), **opt_config['args'])

    # Get the data loader and the permutation function
    data_loader, get_permutation = get_data_loader(args.data_root, USE_CUDA,
                                                   args.batch_size, args.for_outputs)

    # Model training
    model = train_model(model, optimizer, data_loader, args.epochs, args.interval,
                        cur_device, get_permutation)

    # Save model if required
    if args.save_model:
        save_model(model, args.for_outputs)

    # Get representations if required
    if args.get_repr:
        data_type = 'outputs' if args.for_outputs else 'inputs'
        model = model.to(cur_device)
        reprs = get_repr(model, data_loader, cur_device, get_permutation)

        np.savetxt('representation_{}_{}.txt'.format(data_type, TIME_STAMP),
                   reprs.detach().cpu().numpy())

        if data_type == 'outputs':
            print(get_permutation.keywords['indices'])
