from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            # print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   
            # print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # 

        self.gamma = gamma

    def forward(self, inputs, labels):
        
        preds_softmax = F.softmax(inputs)
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
