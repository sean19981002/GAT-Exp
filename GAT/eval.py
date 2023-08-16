from dgl.nn.pytorch import GATConv
from dgl.data import CoraGraphDataset
import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from gat import *
from torchmetrics.classification import BinaryConfusionMatrix


device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

from torchmetrics.classification import MulticlassPrecision 
class evaluation:
    '''
    no need to __init__()
    becuz the train_mask and test_mask will be different in next iteration
    '''
    def Binary_class(self, logits, labels, mask):
        # calculate accuracy and precision of train
        predictions = logits[mask].max(dim=1)[1].to(device)
        correct_predictions = predictions.eq(labels[mask].to(device))
        accuracy = correct_predictions.float().mean().item()

        true_positive = ((predictions == labels[mask].to(device)) & (predictions == 1)).to(device)
        true_positive = true_positive.sum().item()
        positive_predictions = (predictions == 1).to(device)
        positive_predictions = positive_predictions.sum().item()
        precision = true_positive / positive_predictions if positive_predictions > 0 else 0

        return accuracy, precision
    
    def Multi_class(self, model, logits, labels, mask, features):
        # calculate accuracy and precision of testing
        model.eval()
        with torch.no_grad():
            logits = model(features.float().to(device))
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels.to(device))
            acc = correct.item() * 1.0 / len(labels)

            metric = MulticlassPrecision(num_classes=7).to(device= device)
            precision = metric(indices.to(device), labels.to(device)).item()

            return acc, precision
        
    # Multi-clss confusion
    
    # Confusion Matrix
    def confusion_matrix(self, logits, labels, mask):
        predictions = logits[mask].max(dim=1)[1].cpu()
        bcm = BinaryConfusionMatrix()
        conf = bcm(predictions, labels[mask].cpu())
        tn = conf[0, 0].item()
        fp = conf[0, 1].item()
        fn = conf[1, 0].item()
        tp = conf[1, 1].item()
        
        return tp, tn, fp, fn

