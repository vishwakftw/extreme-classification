from datetime import datetime
from argparse import ArgumentParser
from scipy.sparse import csr_matrix
from sklearn import multiclass, svm
from sklearn.externals import joblib

from extreme_classification.loaders import LibSVMLoader
from extreme_classification.hierarchicalXC import HierarchicalXC
from extreme_classification.autoencoders import GenericAutoencoder
from extreme_classification.metrics import precision_at_k, ndcg_score_at_k

import time
import yaml
import torch
import numpy as np
import torch.nn.functional as F


def generate_encodings(data_loader, ae):
    """
    Function to return encoded inputs as a sparse matrix.

    Args:
        data_loader : torch dataloader
        ae: trained autoencoder

    Returns:
        Reconstructed input as a sparse matrix.
    """
    vals = []
    for x, _ in iter(data_loader):
        x = x.to(device=cur_device, dtype=torch.float)
        x = x.squeeze(1)
        vals.append(ae(x))

    vals = torch.cat(vals, dim=0)
    vals = vals.detach().cpu().numpy()
    vals = csr_matrix(vals)
    return vals


TIME_STAMP = datetime.utcnow().isoformat()

parser = ArgumentParser()

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help='Root folder for dataset')
parser.add_argument('--dataset_info', type=str, required=True,
                    help='Dataset information in YAML format')

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
parser.add_argument('--input_ae_dim', type=int, default=-1,
                    help="""Output dimensions of the input encoder.
                    By default, this is -1, meaning no encoding is done""")

# optimizer arguments
parser.add_argument('--optimizer_cfg', type=str, required=True,
                    help='Optimizer configuration in YAML format for Autoencoder model')

# post training arguments
parser.add_argument('--save_model', action='store_true',
                    help='Toggle to save model completely')
parser.add_argument('--k', type=str, default=5,
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

my_hierarchical_XC = HierarchicalXC()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

opt_options = yaml.load(open(args.optimizer_cfg))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loader_kwargs = {}
if USE_CUDA:
    loader_kwargs = {'num_workers': 1, 'pin_memory': True}

loader = LibSVMLoader(args.data_root, yaml.load(open(args.dataset_info)))
len_loader = len(loader)
data_loader = torch.utils.data.DataLoader(loader, batch_size=args.batch_size, shuffle=True,
                                          **loader_kwargs)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input = loader.get_features()

if args.input_ae_dim > 0:
    all_iters = 0
    cur_no = 0
    ae = GenericAutoencoder(input.shape[1], 0.25, args.input_ae_dim)
    ae = ae.to(cur_device)

    optimizer = getattr(torch.optim, opt_options['name'])(ae.parameters(),
                                                          **opt_options['args'])

    INP_REC_LOSS = []

    for epoch in range(args.epochs):
        cur_no = 0
        for x, _ in iter(data_loader):
            x = x.to(device=cur_device, dtype=torch.float)
            cur_no += x.size(0)

            optimizer.zero_grad()
            inp_ae_fp = ae.forward(x)
            loss_inp_rec = F.mse_loss(inp_ae_fp, x)
            loss_inp_rec.backward()
            optimizer.step()

            all_iters += 1
            if all_iters % args.interval == 0:
                print("{} / {} :: {} / {} - INP_REC_LOSS : {}\t"
                      .format(epoch, args.epochs, cur_no, len_loader,
                              round(loss_inp_rec.item(), 5)))
            INP_REC_LOSS.append(loss_inp_rec.item())
    input = generate_encodings(data_loader, ae)

start_time = time.time()
my_hierarchical_XC.train(input, loader.get_classes(), multiclass.OneVsRestClassifier,
                         estimator=svm.SVC(), n_jobs=-1)
print("--- Completed Training in %.5f seconds ---" % (time.time() - start_time))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate the metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

start_time = time.time()
predictions = my_hierarchical_XC.predict(input)
print("--- Completed Predicting in %.5f seconds ---" % (time.time() - start_time))

pred_y = predictions.toarray()
actual_y = loader.get_classes().toarray()

k = args.k

p_at_k = [precision_at_k(actual_y[i], pred_y[i], k) for i in range(len(pred_y))]
ndcg_at_k = [ndcg_score_at_k(actual_y[i], pred_y[i], k) for i in range(len(pred_y))]

print("Precision at {0} = {1}".format(k, np.mean(p_at_k)))
print("NDCG at {0} = {1}".format(k, np.mean(ndcg_at_k)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.save_model:
    joblib.dump(my_hierarchical_XC, 'trained_hierarchial_model_{}.sav'.format(TIME_STAMP))
