from sklearn import multiclass, svm
from datetime import datetime
from functools import partial
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

from extreme_classification.hierarchicalXC import HierarchicalXC
from extreme_classification.loaders import LibSVMLoader
from extreme_classification.metrics import precision_at_k, ndcg_score_at_k

import yaml
import numpy as np

TIME_STAMP = datetime.utcnow().isoformat()

parser = ArgumentParser()

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help='Root folder for dataset')
parser.add_argument('--dataset_info', type=str, required=True,
                    help='Dataset information in YAML format')

# training configuration arguments
parser.add_argument('--seed', type=int, default=None,
                    help='Manually set the seed for the experiments for reproducibility')
parser.add_argument('--plot', action='store_true',
                    help='Option to plot the loss variation over iterations')

# post training arguments
parser.add_argument('--k', type=str, default=5,
                    help='k for Precision at k and NDCG at k')

# parse the arguments
args = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

my_hierarchical_XC = HierarchicalXC()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loader_kwargs = {}
loader = LibSVMLoader(args.data_root, yaml.load(open(args.dataset_info)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

my_hierarchical_XC.train(loader, multiclass.OneVsRestClassifier)
predictions = my_hierarchical_XC.predict(loader.get_data()[0])
print(predictions.toarray())
