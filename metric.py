import torch


def topk_precision(y_pred: torch.Tensor, y: torch.Tensor, k: int = 2):
    # y_pred: [B x C] logits
    # y: [B x C] multi-hot
    B=y_pred.shape[0]
    values,indices=torch.topk(y_pred,k,sorted=True)
    pred_multi_hot=(y_pred>=values[:,[-1]]).float()
    return (pred_multi_hot*y).sum()/B/k
