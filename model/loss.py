import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss, TripletMarginLoss


def nll_loss(output, target):
    # import pdb; pdb.set_trace()
    return F.nll_loss(output, target)

def mse_loss(output, target):
    loss = MSELoss()
    return loss(output, target.double())

def retrieval_loss2(output, target, no_negatives: int):
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
