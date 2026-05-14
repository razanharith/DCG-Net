import torch
import copy

# SR : Segmentation Result
# GT : Ground Truth

#thresholdlist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
thresholdlist = [0.5]

def get_accuracy(SR, GT, thresholdlist=thresholdlist):
    # Accuracy
    acc = 0
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_accuracy - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return 0.0
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1
        
        corr = SR_copy.eq(GT_copy).sum()
        tensor_size = SR_copy.numel()  # Use numel() for safer counting
        acc_copy = float(corr)/float(tensor_size)
        
        if acc_copy > acc:
            acc = copy.copy(acc_copy)
        
    return acc

def get_Recall(SR, GT, thresholdlist=thresholdlist):
    # Recall == Sensitivity
    RC = 0
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_Recall - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return 0.0
            
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1

        # TP : True Positive
        # FN : False Negative
        TP = torch.sum((SR_copy==1)&(GT_copy==1))
        FN = torch.sum((SR_copy==0)&(GT_copy==1))
        
        RC_copy = float(TP)/(float(TP+FN) + 1e-8)
        
        if RC_copy > RC:
            RC = copy.copy(RC_copy)
        
    return RC

def get_specificity(SR, GT, thresholdlist=thresholdlist):
    SP = 0
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_specificity - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return 0.0
            
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1

        # TN : True Negative
        # FP : False Positive
        TN = torch.sum((SR_copy==0)&(GT_copy==0))
        FP = torch.sum((SR_copy==1)&(GT_copy==0))
        
        SP_copy = float(TN)/(float(TN+FP) + 1e-8)
        
        if SP_copy > SP:
            SP = copy.copy(SP_copy)
        
    return SP

def get_Precision(SR, GT, thresholdlist=thresholdlist):
    PC = 0
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_Precision - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return 0.0
            
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1

        # TP : True Positive
        # FP : False Positive
        TP = torch.sum((SR_copy==1)&(GT_copy==1))
        FP = torch.sum((SR_copy==1)&(GT_copy==0))
        
        PC_copy = float(TP)/(float(TP+FP) + 1e-8)
        
        if PC_copy > PC:
            PC = copy.copy(PC_copy)
        
    return PC

def get_F1(SR,GT,thresholdlist=thresholdlist):
    # F1-Score == Dice Score
    RC = get_Recall(SR,GT,thresholdlist=thresholdlist)
    PC = get_Precision(SR,GT,thresholdlist=thresholdlist)

    F1 = 2*RC*PC/(RC+PC + 1e-8)

    return F1

def get_mIoU(SR, GT, thresholdlist=thresholdlist):
    # mIoU : Mean of Intersection over Union (Jaccard Index)
    IoU1 = 0
    mIoU = 0
    thresh = 0.5  # Initialize thresh with default value
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_mIoU - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return (0.0, 0.0)
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1
    
        # IoU1 of Foreground
        Inter1 = torch.sum((SR_copy==1)&(GT_copy==1))
        Union1 = torch.sum(SR_copy==1) + torch.sum(GT_copy==1) - Inter1
        IoU1_copy = float(Inter1)/(float(Union1) + 1e-8)
        if IoU1_copy > IoU1:
            IoU1 = copy.copy(IoU1_copy)
            
        # IoU2 of Background
        Inter2 = torch.sum((SR_copy==0)&(GT_copy==0))
        Union2 = torch.sum(SR_copy==0) + torch.sum(GT_copy==0) - Inter2
        IoU2 = float(Inter2)/(float(Union2) + 1e-8)
            
        mIoU_copy = (IoU1 + IoU2) / 2
        
        if mIoU_copy > mIoU:
            mIoU = copy.copy(mIoU_copy)
            thresh = threshold
        
    # Return only IoU1 and mIoU to match expected format in solver.py
    return IoU1, mIoU

def get_DC(SR, GT, thresholdlist=thresholdlist):
    # DC : Dice Coefficient
    DC = 0
    
    # Check for size mismatch and resize if necessary
    if SR.size() != GT.size():
        print(f"Warning: Size mismatch in get_DC - SR: {SR.size()}, GT: {GT.size()}. Resizing SR.")
        try:
            GT = GT.view(-1)  # Flatten GT
            SR = SR.view(-1)  # Flatten SR
            
            # Ensure they're the same size by truncating the longer one
            min_size = min(SR.size(0), GT.size(0))
            SR = SR[:min_size]
            GT = GT[:min_size]
        except Exception as e:
            print(f"Error handling tensor size mismatch: {e}")
            return 0.0
    
    for threshold in thresholdlist:
        SR_copy = copy.deepcopy(SR)
        GT_copy = copy.deepcopy(GT)
        SR_copy[SR_copy < threshold] = 0
        SR_copy[SR_copy > threshold] = 1
        GT_copy[GT_copy < 0.8] = 0
        GT_copy[GT_copy > 0.8] = 1
    
        Inter = torch.sum((SR_copy==1)&(GT_copy==1))
        Union = torch.sum(SR_copy==1)+torch.sum(GT_copy==1)
        DC_copy = float(2*Inter)/(float(Union) + 1e-8)
        
        if DC_copy > DC:
            DC = copy.copy(DC_copy)
    
    return DC

