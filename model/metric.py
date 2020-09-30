import torch

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    obtained from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

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
