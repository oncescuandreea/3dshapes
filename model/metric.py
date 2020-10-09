import numpy as np
import torch
from utils.util import pairwise_distances, similarity_matrix_01l2

def accuracy_tot(output, target, no_tasks):
    correct = 0
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        for i in range(0, no_tasks):
            output_task = output[i]
            target_task = target[:, i]
            pred = torch.argmax(output_task, dim=1)
            assert pred.shape[0] == len(target_task)
            correct += torch.sum(pred == target_task).item()
    return correct / (6 * len(target))

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_retrieval(output, target):
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        batch_len = len(target)
        dist = pairwise_distances(output, target)
        min_dist_location = dist.argmin(1)
        tensor_idx = torch.tensor(range(0, batch_len)).cuda()
        bool_correct_min_location = (min_dist_location == tensor_idx).tolist()
        correct = bool_correct_min_location.count(True)
    return correct / batch_len

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def compute_metric(output, target):
    """Started from:
    https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loss.py#L30
    """
    distance_matrix = pairwise_distances(output, target)
    x = similarity_matrix_01l2(distance_matrix).cpu().detach().numpy()
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1

    return metrics
