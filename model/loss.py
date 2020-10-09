import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MSELoss, TripletMarginLoss
from utils.util import pairwise_distances, similarity_matrix_01l2


def similarity_matrix_cos(a: th.Tensor, b: th.Tensor, dim: int = -1):
    """Function calculated the cosine similarity matrix to be used
    with the MaxMarginRankingLoss class"""
    return th.cosine_similarity(a.unsqueeze(-2), b, dim=dim)


class MaxMarginRankingLoss(nn.Module):
    """Copied from:
    https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loss.py#L30
    """
    def __init__(self, margin=1):
        super(MaxMarginRankingLoss, self).__init__()
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        # import pdb; pdb.set_trace()
        n = x.size()[0]
        
        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0) 

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)
       
        x2 = th.cat((x2, x3), 0)
         
        max_margin = F.relu(self.margin - (x1 - x2))
        # import pdb; pdb.set_trace()
        return max_margin.mean()

def nll_loss(output, target):
    # import pdb; pdb.set_trace()
    return F.nll_loss(output, target)

def mse_loss(output, target):
    loss = MSELoss()
    return loss(output, target.double())

def retrieval_loss2(output, target, no_negatives=30):
    """Function returns triplet loss obtained from comparing the anchor
    with the corresponding postivie label and with another no_negatives
    negative examples
    Inputs:
        output: tensor containing image embedding
        target: tensor containing text embedding
        no_negatives: integer containing number of negative examples"""
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    batch_len = len(target)
    batch_loss = 0
    negative_idx = np.random.randint(0, batch_len, np.min([batch_len, no_negatives]))

    total_examples = no_negatives * batch_len
    batch_example_dim = len(output[0])
    # import pdb; pdb.set_trace()
    for k, output_example in enumerate(output):
        output_example_reshaped = output_example.reshape([1, batch_example_dim])
        for idx in negative_idx:
            if idx != k:
                batch_loss += triplet_loss(output_example_reshaped, target[k], target[idx])
            else:
                total_examples -= 1
    return batch_loss / total_examples

def retrieval_loss_full_matrix(output, target):
    confusion_matrix = pairwise_distances(output, target)
    # confusion_matrix = similarity_matrix_cos(output, target)
    confusion_matrix = similarity_matrix_01l2(confusion_matrix)
    max_margin_loss = MaxMarginRankingLoss()
    return max_margin_loss(confusion_matrix)


