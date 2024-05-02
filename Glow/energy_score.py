import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.metrics as sk
import logging 
# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)


from build_dataset import build_or_get_dataset, get_dataloader
from model import Glow
device = torch.device("cuda")
from config import get_config

config = get_config()
path = "./Glow/checkpoints/glow_checkpoint_10.pt"

num_classes = 10
image_shape = (32, 32, 3)
model = Glow(image_shape,config['hidden_channels'],config['K'],config['L'],config['actnorm_scale'],
    config['flow_permutation'],config['flow_coupling'],config['LU_decomposed'],num_classes,config['learn_top'],config['y_condition'],
)
model.load_state_dict(torch.load(path)['model'])
model.set_actnorm_init()
model = model.to(device)
model = model.eval()

recall_level_default = 0.95

log_dir = './Glow/logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{log_dir}/results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

_, testset_in, classes = build_or_get_dataset('cifar10', '../data',task_generation=True)
_, testset_out, _= build_or_get_dataset('svhn', '../data',task_generation=True)
testloader_in = get_dataloader(testset_in, batch_size = config["batch_size"], shuffle= False)
testloader_out = get_dataloader(testset_out, batch_size = config["batch_size"], shuffle= False)

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps     
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true))) 

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr

def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)
    logger.info(f"{method_name} : FPR{int(100 * recall_level)} {100*fpr:2f} AUROC {100*auroc:.2f} AUPR {100*aupr:.2f}")

def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    logger.info(f"{method_name} : FPR{int(100 * recall_level)} {100*fpr:2f} AUROC {100*auroc:.2f} AUPR {100*aupr:.2f}")

ood_num_examples = len(testset_in) // 5
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, score='MSP',in_dist=False, T=1):
    _score = []
    _right_score = []
    _wrong_score = []
    progress_bar = tqdm(loader, desc=f'Testing {score} in_dist {in_dist}', leave=False)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            if batch_idx >= ood_num_examples // config['batch_size'] and in_dist is False:
                break
            data = data.to(device)
            target = target.to(device)

            
            _, _, output = model(data, y_onehot=target)
            smax = to_np(F.softmax(output, dim=1))
            if score == 'energy':
                _score.append(-to_np((T*torch.logsumexp(output /T, dim=1))))
            else: 
                _score.append(-np.max(smax, axis=1))
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.cpu().numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()
    
in_score_msp, right_score, wrong_score = get_ood_scores(testloader_in, in_dist=True, score='MSP')
in_score_energy, right_score, wrong_score = get_ood_scores(testloader_in, in_dist=True, score='energy')

def plot_histogram(in_score,out_score, score):
    plt.figure(figsize=(10, 6))
    sns.histplot(in_score, bins=200, kde=True, label='CIFAR-10', color='blue', alpha=0.7, stat='density')
    sns.histplot(out_score, bins=200, kde=True, label='SVHN', color='red', alpha=0.7, stat='density')
    plt.xlabel(score)
    plt.ylabel('Density')
    plt.title(f'{score} Score Distribution: CIFAR-10 vs. SVHN')
    plt.legend()
    plt.savefig(f'./Glow/fig/{score}_e10_histogram.png')
    plt.show()

def get_and_print_results(ood_loader,in_score, num_to_avg=1, score='MSP'):
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, score=score)
        measures = get_measures(-in_score, -out_score)
        plot_histogram(-in_score, -out_score, score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, score)


get_and_print_results(testloader_out, in_score_msp, score= 'MSP')
get_and_print_results(testloader_out, in_score_energy, score= 'energy')