from datetime import datetime
from argparse import ArgumentParser
from sklearn import multiclass, svm
from sklearn.externals import joblib

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
parser.add_argument('--save_model', action='store_true', default=False,
                    help='Option to save the model completely')
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

my_hierarchical_XC.train(loader, multiclass.OneVsRestClassifier, estimator=svm.SVC(gamma="scale"))
predictions = my_hierarchical_XC.predict(loader.get_data()[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate the metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pred_y = predictions.toarray()
actual_y = []
for x, y in iter(loader):
    actual_y.append(y.numpy().reshape(-1))

k = args.k

p_at_k = [precision_at_k(actual_y[i], pred_y[i], k) for i in range(len(pred_y))]
ndcg_at_k = [ndcg_score_at_k(actual_y[i], pred_y[i], k) for i in range(len(pred_y))]

print("Precision at {0} = {1}".format(k, np.mean(p_at_k)))
print("NDCG at {0} = {1}".format(k, np.mean(ndcg_at_k)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.save_model is not None:
    joblib.dump(my_hierarchical_XC, 'trained_hierarchial_model_{}.sav'.format(TIME_STAMP))
