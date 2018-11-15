from datetime import datetime
from functools import partial
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

from extreme_classification.neuXC import NeuralXC
from extreme_classification.loaders import LibSVMLoader
from extreme_classification.metrics import precision_at_k, ndcg_score_at_k

import yaml
import torch
import numpy as np
import torch.nn.functional as F


def weights_init(mdl, scheme):
    """
    Function to initialize weights

    Args:
        mdl : Module whose weights are going to modified
        scheme : Scheme to use for weight initialization
    """
    if isinstance(mdl, torch.nn.Linear):
        func = getattr(torch.nn.init, scheme + '_')  # without underscore is deprecated
        func(mdl.weight)


TIME_STAMP = datetime.utcnow().isoformat()

parser = ArgumentParser()

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help='Root folder for dataset')
parser.add_argument('--dataset_info', type=str, required=True,
                    help='Dataset information in YAML format')

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
parser.add_argument('--init_scheme', type=str, default='default',
                    choices=['xavier_uniform', 'kaiming_uniform', 'default'])

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
parser.add_argument('--plot', action='store_true',
                    help='Option to plot the loss variation over iterations')

# optimizer arguments
parser.add_argument('--optimizer_cfg', type=str, required=True,
                    help='Optimizer configuration in YAML format for NeuralXC model')

# post training arguments
parser.add_argument('--save_model', type=str, default=None,
                    choices=['all', 'inputAE', 'outputAE', 'regressor'], nargs='+',
                    help='Options to save the model partially or completely')
parser.add_argument('--k', type=int, default=5,
                    help='k for Precision at k and NDCG at k')

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
if args.init_scheme != 'default':
    my_neural_XC.apply(partial(weights_init, scheme=args.init_scheme))
my_neural_XC = my_neural_XC.to(cur_device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

opt_options = yaml.load(open(args.optimizer_cfg))
optimizer = getattr(torch.optim, opt_options['name'])(my_neural_XC.parameters(),
                                                      **opt_options['args'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loader_kwargs = {}
if USE_CUDA:
    loader_kwargs = {'num_workers': 1, 'pin_memory': True}

loader = LibSVMLoader(args.data_root, yaml.load(open(args.dataset_info)))
len_loader = len(loader)
data_loader = torch.utils.data.DataLoader(loader, batch_size=args.batch_size, shuffle=True,
                                          **loader_kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_iters = 0
ALPHA_INPUT = args.input_ae_loss_weight
ALPHA_OUTPUT = args.output_ae_loss_weight
K = args.k
INP_REC_LOSS = []
OTP_REC_LOSS = []
CLASS_LOSS = []
AVG_P_AT_K = []
AVG_NDCG_AT_K = []

for epoch in range(args.epochs):
    cur_no = 0
    for x, y in iter(data_loader):
        x = x.to(device=cur_device, dtype=torch.float)
        y = y.to(device=cur_device, dtype=torch.float)
        cur_no += x.size(0)

        optimizer.zero_grad()
        inp_ae_fp, out_ae_fp, reg_fp = my_neural_XC(x, y)

        # This is a custom that we will be using to backprop. It has three components:
        # 1. Reconstruction error of input
        # 2. Reconstruction error of output
        # 3. Classification (Binary cross entropy) of input-output
        # The first two are weighted using ALPHA_INPUT and ALPHA_OUTPUT.
        loss_inp_rec = F.mse_loss(inp_ae_fp, x)
        loss_otp_rec = F.mse_loss(out_ae_fp, y)
        loss_class = F.binary_cross_entropy(reg_fp, y)
        net_loss = ALPHA_INPUT * loss_inp_rec + ALPHA_OUTPUT * loss_otp_rec + loss_class
        net_loss.backward()
        optimizer.step()
        all_iters += 1
        if all_iters % args.interval == 0:
            print("{} / {} :: {} / {} - INP_REC_LOSS : {}\tOTP_REC_LOSS : {}\tCLASS_LOSS : {}"
                  .format(epoch, args.epochs, cur_no, len_loader,
                          round(loss_inp_rec.item(), 5), round(loss_otp_rec.item(), 5),
                          round(loss_class.item(), 5)))
        INP_REC_LOSS.append(loss_inp_rec.item())
        OTP_REC_LOSS.append(loss_otp_rec.item())
        CLASS_LOSS.append(loss_class.item())

    pred_y = []
    actual_y = []
    for x, y in iter(data_loader):
        x = x.to(device=cur_device, dtype=torch.float)

        pred_y.append(my_neural_XC.predict(x).detach().cpu().numpy())
        actual_y.append(y.numpy())

    pred_y = np.vstack(pred_y)
    actual_y = np.vstack(actual_y)
    p_at_k = [precision_at_k(actual_y[i], pred_y[i], K) for i in range(len(pred_y))]
    ndcg_at_k = [ndcg_score_at_k(actual_y[i], pred_y[i], K) for i in range(len(pred_y))]
    print("{0} / {1} :: Precision at {2}: {3}\tNDCG at {2}: {4}"
          .format(epoch, args.epochs, K, np.mean(p_at_k), np.mean(ndcg_at_k)))
    AVG_P_AT_K.append(np.mean(p_at_k))
    AVG_NDCG_AT_K.append(np.mean(ndcg_at_k))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot graphs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if args.plot:
    fig = plt.figure(figsize=(9, 18))
    gridspec = gs.GridSpec(4, 6, figure=fig)
    gridspec.tight_layout(fig)
    ax1 = plt.subplot(gridspec[0, :2])
    ax2 = plt.subplot(gridspec[0, 2:4])
    ax3 = plt.subplot(gridspec[0, 4:])
    ax4 = plt.subplot(gridspec[1:3, 1:5])
    ax5 = plt.subplot(gridspec[3, :3])
    ax6 = plt.subplot(gridspec[3, 3:])

    ax1.plot(list(range(1, all_iters + 1)), INP_REC_LOSS, 'r', linewidth=2.0)
    ax1.set_title('Input reconstruction loss : weight = {}'.format(ALPHA_INPUT))
    ax2.plot(list(range(1, all_iters + 1)), OTP_REC_LOSS, 'g', linewidth=2.0)
    ax2.set_title('Output reconstruction loss : weight = {}'.format(ALPHA_OUTPUT))
    ax3.plot(list(range(1, all_iters + 1)), CLASS_LOSS, 'b', linewidth=2.0)
    ax3.set_title('Classification loss')
    ax4.plot(list(range(1, all_iters + 1)),
             [ALPHA_INPUT * irl + ALPHA_OUTPUT * orl + cl
              for (irl, orl, cl) in zip(INP_REC_LOSS, OTP_REC_LOSS, CLASS_LOSS)],
             'k', linewidth=3.0)
    ax4.set_title('All losses')
    ax5.plot(list(range(1, args.epochs + 1)), AVG_P_AT_K, 'g', linewidth=2.0)
    ax5.set_title('Average Precision at {} (over all datapoints) with epochs'.format(K))
    ax6.plot(list(range(1, args.epochs + 1)), AVG_NDCG_AT_K, 'b', linewidth=2.0)
    ax6.set_title('Average NDCG at {} (over all datapoints) with epochs'.format(K))
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.save_model is not None:
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
