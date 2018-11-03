from datetime import datetime
from argparse import ArgumentParser
from .neu_XC import NeuralXC
from ..loaders import LibSVMLoader

import yaml
import torch
import torch.nn.functional as F


TIME_STAMP = datetime.utcnow().isoformat()

parser = ArgumentParser()

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help='Root folder for dataset')

# architecture arguments
parser.add_argument('--input_encoder_cfg', type=str, required=True,
                    help='Input Encoder architecture configuration in YAML format')
parser.add_argument('--input_decoder_cfg', type=str, required=True,
                    help='Input Decoder architecture configuration in YAML format')
parser.add_argument('--output_encoder_cfg', type=str, required=True,
                    help='Output Encoder architecture configuration in YAML format')
parser.add_argument('--output_decoder_cfg', type=str, required=True,
                    help='Output Decoder architecture configuration in YAML format')
parser.add_argument('--regressor_cfg', type=str, required=True,
                    help='Regressor architecture configuration in YAML format')

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
parser.add_argument('--input_ae_loss_weight', type=float, default=1.,
                    help='Weight to give the input autoencoder loss in the entire loss')
parser.add_argument('--output_ae_loss_weight', type=float, default=1.,
                    help='Weight to give the output autoencoder loss in the entire loss')

# optimizer arguments
parser.add_argument('--optimizer_cfg', type=str, required=True,
                    help='Optimizer configuration in YAML format for NeuralXC model')

# post training arguments
parser.add_argument('--save_model', type=str, default='all',
                    choices=['all', 'inputAE', 'outputAE', 'regressor'], nargs='+',
                    help='Options to save the model partially or completely')

# parse the arguments
args = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CUDA Capability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cur_device = torch.device(args.device)
USE_CUDA = cur_device.type == 'cuda'
if USE_CUDA and not torch.cuda.is_available():
    raise ValueError("You can't use CUDA if you don't have CUDA")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reproducibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.seed is not None:
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_enc_cfg = yaml.load(open(args.input_encoder_cfg))
input_dec_cfg = yaml.load(open(args.input_decoder_cfg))
output_enc_cfg = yaml.load(open(args.output_encoder_cfg))
output_dec_cfg = yaml.load(open(args.output_decoder_cfg))
regress_cfg = yaml.load(open(args.regressor_cfg))

my_neural_XC = NeuralXC(input_enc_cfg, input_dec_cfg, output_enc_cfg, output_dec_cfg, regress_cfg)
my_neural_XC = my_neural_XC.to(cur_device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

opt_options = yaml.load(open(args.input_ae_optimizer_cfg))
optimizer = getattr(torch.optim, opt_options['name'])(my_neural_XC.parameters(),
                                                      **opt_options['args'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loader_kwargs = {}
if USE_CUDA:
    loader_kwargs = {'num_workers': 1, 'pin_memory': True}

loader = LibSVMLoader(args.data_root)
data_loader = torch.utils.data.DataLoader(loader, batch_size=args.batch_size, shuffle=True,
                                          **loader_kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_iters = 0
ALPHA_INPUT = args.input_ae_loss_weight
ALPHA_OUTPUT = args.output_ae_loss_weight

for i in range(args.epochs):
    for x, y in iter(data_loader):
        x = x.to(device=cur_device, dtype=torch.float)
        y = y.to(device=cur_device, dtype=torch.float)

        optimizer.zero_grad()
        inp_ae_fp, out_ae_fp, reg_fp = my_neural_XC(x, y)

        # This is a custom that we will be using to backprop. It has three components:
        # 1. Reconstruction error of input
        # 2. Reconstruction error of output
        # 3. Classification (Binary cross entropy) of input-output
        # The first two are weighted using ALPHA_INPUT and ALPHA_OUTPUT.
        loss_inp_rec = ALPHA_INPUT * F.mse_loss(inp_ae_fp, x)
        loss_otp_rec = ALPHA_OUTPUT * F.mse_loss(out_ae_fp, y)
        loss_class = F.binary_cross_entropy(reg_fp, y)
        net_loss = loss_inp_rec + loss_otp_rec + loss_class
        net_loss.backward()
        optimizer.step()
        all_iters += 1
        if all_iters % args.interval == 0:
            print("{} / {} - INP_REC_LOSS : {}\tOTP_REC_LOSS : {}\tCLASS_LOSS : {}".format(
                  i, args.epochs, round(loss_inp_rec.item(), 5), round(loss_otp_rec.item(), 5),
                  round(loss_class.item(), 5)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if 'inputAE' in args.save_model or 'all' in args.save_model:
    torch.save(my_neural_XC.input_encoder.to('cpu'),
               'trained_input_encoder_{}.pt'.format(TIME_STAMP))
    torch.save(my_neural_XC.input_decoder.to('cpu'),
               'trained_input_decoder_{}.pt'.format(TIME_STAMP))

if 'outputAE' in args.save_model or 'all' in args.save_model:
    torch.save(my_neural_XC.output_encoder.to('cpu'),
               'trained_output_encoder_{}.pt'.format(TIME_STAMP))
    torch.save(my_neural_XC.output_decoder.to('cpu'),
               'trained_output_decoder_{}.pt'.format(TIME_STAMP))

if 'regressor' in args.save_model or 'all' in args.save_model:
    torch.save(my_neural_XC.regressor.to('cpu'),
               'trained_regressor_{}.pt'.format(TIME_STAMP))
