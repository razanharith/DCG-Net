import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, threshold, weight=None, size_average=True):
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
    def __init__(self, threshold, weight=None, size_average=True):
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
    def __init__(self, threshold, weight=None, size_average=True):
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
        
def displayfigures(results, result_path, report_file, dataset, model):
    import os
    # Create directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    for i in range(len(results)):
        plt.Figure()
        plt.plot(results[i][1], marker='o', markersize=3, label="Train "+results[i][0])
        plt.plot(results[i][2], marker='o', markersize=3, label="Val "+results[i][0])
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.ylabel(results[i][0]+"%")
        if results[i][0] != "Loss":
            plt.ylim(0,100)
        plt.savefig(result_path+report_file+'_'+dataset+'_'+model+'_'+results[i][0]+'_results.png')
        plt.close()

def PRC(PC, RC, result_path, report_name):
    # Ensure that RC and PC are lists of iterables
    if not isinstance(RC, (list, np.ndarray)) or not len(RC):
        raise ValueError("RC must be a non-empty list or numpy array.")
    if not isinstance(PC, (list, np.ndarray)) or not len(PC):
        raise ValueError("PC must be a non-empty list or numpy array.")
    
    # Convert to lists of lists if they are not already
    RC = [list([rc]) if not isinstance(rc, (list, np.ndarray)) else rc for rc in RC]
    PC = [list([pc]) if not isinstance(pc, (list, np.ndarray)) else pc for pc in PC]
    
    # Transposing RC and PC
    RC = list(map(list, zip(*RC)))
    PC = list(map(list, zip(*PC)))
    
    # Calculate mean values
    RC1 = [np.mean(r) for r in RC]
    PC1 = [np.mean(p) for p in PC]

    # Avoid negative AUC by flipping arrays
    PC = np.fliplr([PC1])[0]
    RC = np.fliplr([RC1])[0]

    # Calculate the area under the curve
    AUC_PC_RC = np.trapz(PC, RC)
    print("\nArea under Precision-Recall curve: " + str(AUC_PC_RC))

    # Plotting
    plt.figure()
    plt.plot(RC, PC, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_PC_RC)
    plt.title('Precision - Recall curve')
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(result_path + report_name + '_Precision_recall.png')

    return RC, PC