from datetime import datetime
from argparse import ArgumentParser
from extreme_classification.loaders import LibSVMLoader
from extreme_classification.autoencoders import Autoencoder as AE
from extreme_classification.clusterings import CoOccurrenceAgglomerativeClustering as CAAC

import sys
import yaml
import torch


parser = ArgumentParser()

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help='Root folder for dataset')
parser.add_argument('--outputs', action='store_true',
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
args = parser.parse_args()


using_cuda = args.device.find('cuda')
if not torch.cuda.is_available() and using_cuda:
    print("Attempting to use CUDA when CUDA is unavailable!!")
    sys.exit(1)

device = torch.device(args.device)

if args.seed is not None:
    torch.manual_seed(args.seed)
    if using_cuda:
        torch.cuda.manual_seed(args.seed)

loader = LibSVMLoader(args.data_root)
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if using_cuda else {}
data_loader = torch.utils.data.DataLoader(loader, batch_size=args.batch_size, shuffle=True,
                                          **loader_kwargs)

model = AE(yaml.load(open(args.encoder_cfg)), yaml.load(open(args.decoder_cfg))).to(device)

opt_config = yaml.load(open(args.optimizer_cfg))
optimizer = getattr(torch.optim, opt_config['name'])(model.parameters(), **opt_config['args'])

for_outputs = args.for_outputs
interval = args.interval

if for_outputs:
    agglo_clusters = CAAC(loader)
    get_permutation = agglo_clusters.get_permutation_lambda

for epoch in range(0, args.epochs):
    model.train()
    for batch_idx, (inp, out) in enumerate(data_loader):
        if for_outputs:
            data = get_permutation(out).to(device=device, dtype=torch.float32)
        else:
            data = inp.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % interval == 0:
            print('Training epoch {} : Loss = {}'.format(epoch, loss.item()))

# Transfer the model to CPU before saving
if args.save_model:
    split_str = 'outputs' if for_outputs else 'inputs'
    torch.save(model.to('cpu'),
               'trained_model_{}_{}.pt'.format(split_str, datetime.utcnow().isoformat()))
