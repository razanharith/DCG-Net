
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


class DiceLoss(nn.Module):
    def __init__(self, threshold):
        super(DiceLoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-8): 
        
        SR = SR.view(-1)
        GT = GT.view(-1)
        Inter = torch.sum((SR>self.threshold)&(GT>0.8))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>0.8)
        Dice = float(2.*Inter)/(float(Union) + smooth)
        
        return 1 - Dice

class IoULoss(nn.Module):
    def __init__(self, threshold):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-8):
        SR = SR.view(-1)
        GT = GT.view(-1) 
        Inter = torch.sum((SR>self.threshold)&(GT>0.8))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>0.8) - Inter
        IoU = float(Inter)/(float(Union) + smooth)
                
        return 1 - IoU

class mIoULoss(nn.Module):
    def __init__(self, threshold):
        super(mIoULoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, SR, GT, smooth=1e-8):
        SR = SR.view(-1)
        GT = GT.view(-1)
        
        # IoU of Foreground
        Inter1 = torch.sum((SR>self.threshold)&(GT>0.8))
        Union1 = torch.sum(SR>self.threshold) + torch.sum(GT>0.8) - Inter1
        IoU1 = float(Inter1)/(float(Union1) + smooth)

        # IoU of Background
        Inter2 = torch.sum((SR<self.threshold)&(GT<0.8))
        Union2 = torch.sum(SR<self.threshold) + torch.sum(GT<0.8) - Inter2
        IoU2 = float(Inter2)/(float(Union2) + smooth)

        mIoU = (IoU1 + IoU2) / 2
                
        return 1 - mIoU

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

import numpy as np
import matplotlib.pyplot as plt

def PRC(PC, RC, result_path, report_name):
    # Ensure RC and PC are lists of lists (or similar structure) for zip to work
    if not RC or not isinstance(RC[0], (list, tuple)):
        raise ValueError("RC must be a list of lists for zip to work.")
    if not PC or not isinstance(PC[0], (list, tuple)):
        raise ValueError("PC must be a list of lists for zip to work.")

    RC1 = []
    PC1 = []

    # Transpose RC and PC
    RC = list(map(list, zip(*RC)))
    PC = list(map(list, zip(*PC)))

    for i in range(len(RC)):
        if len(RC[i]) > 0:
            RC1.append(np.sum(RC[i]) / len(RC[i]))

    for i in range(len(PC)):
        if len(PC[i]) > 0:
            PC1.append(np.sum(PC[i]) / len(PC[i]))

    # Flip the arrays to avoid negative AUC
    PC = np.fliplr([PC1])[0]
    RC = np.fliplr([RC1])[0]

    # Calculate the Area Under the Curve (AUC)
    AUC_PC_RC = np.trapz(PC, RC)
    print("\nArea under Precision-Recall curve: " + str(AUC_PC_RC))

    # Plotting
    plt.figure()
    plt.plot(RC, PC, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_PC_RC)
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(result_path + report_name + '_Precision_recall.png')
    plt.close()  # Close the plot to avoid display during batch processing

    return RC, PC