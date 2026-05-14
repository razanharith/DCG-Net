import os
import numpy as np
import time
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import re
from ptflops import get_model_complexity_info

# Optional imports with error handling
try:
    from FAT_Net import FAT_Net
except ImportError:
    FAT_Net = None

# PRIMARY MODEL: DCGNet
from DCGNet import DCGNet, DCGNetLoss
try:
    from EADSNet import EADSNet
except ImportError:
    EADSNet = None
try:
    from EADSNetV2 import EADSNet as EADSNetV2
except ImportError:
    EADSNetV2 = None

import torchvision
from UNet import U_Net
try:
    from SegNet import SegNet
except ImportError:
    SegNet = None
from torch import optim
# from torch.autograd import Variable # This import is present but Variable is not used, can be removed.
import torch.nn.functional as F
from evaluation import *
# from swin_transformer import SwinTransformer # Commented out
# from network import U_Net # Commented out
import cv2
import segmentation_models_pytorch as smp
import csv
from misc import *
import os
import argparse
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter
from network import R2U_Net, AttU_Net, R2AttU_Net
# Optional imports with error handling
try:
    from LM_Net import LM_Net 
except ImportError:
    LM_Net = None
try:
    from LM_Net_dw import LM_Net as EnhancedLM_Net
except ImportError:
    EnhancedLM_Net = None
try:
    from LM_Net_eem import LM_Net_eem
except ImportError:
    LM_Net_eem = None
try:
    from LM_Net_PFM import LM_Net_PFM
except ImportError:
    LM_Net_PFM = None
try:
    from LM_Net_PDAM import LM_Net_PDAM
except ImportError:
    LM_Net_PDAM = None

try:
    from CAM_net import CAM_Net
except ImportError:
    CAM_Net = None
try:
    from CAM_net_baseline import CAM_Net_Baseline
except ImportError:
    CAM_Net_Baseline = None
try:
    from CAM_net_msm import CAM_Net_MSM
except ImportError:
    CAM_Net_MSM = None
try:
    from CAM_net_fem import CAM_Net_FEM
except ImportError:
    CAM_Net_FEM = None
try:
    from CAM_net_aspp import CAM_Net_ASPP
except ImportError:
    CAM_Net_ASPP = None
try:
    from CAM_net_eea import CAM_Net_EEA
except ImportError:
    CAM_Net_EEA = None

try:
    from vmunet import VMUNet
except ImportError:
    VMUNet = None
try:
    from vmunetv2 import VMUNetV2
except ImportError:
    VMUNetV2 = None
try:
    from vmamba import VSSM 
except ImportError:
    VSSM = None
try:
    from Shufflenetv2 import ShuffleNetV2Seg,shufflenets
except ImportError:
    ShuffleNetV2Seg = shufflenets = None
try:
    from Mobilenetv1 import MobileNetV1
except ImportError:
    MobileNetV1 = None
try:
    from Mobilenetv2 import MobileNetV2
except ImportError:
    MobileNetV2 = None
try:
    from LightMUNet_wrapper import LightMUNet_2D
except ImportError:
    LightMUNet_2D = None
try:
    from Mobilenetv3 import MobileNetV3Seg
except ImportError:
    MobileNetV3Seg = None
try:
    from Mobilenetv4 import mobilenetv4_conv_small
except ImportError:
    mobilenetv4_conv_small = None


writer = SummaryWriter('mylogdir')


class Solver(object):
    def __init__(self, config, model, train_loader, valid_loader, test_loader):

        # Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Hyper-parameters
        self.lr = config.lr
        self.optimizer_type = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.01

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        self.model_type = model
        self.dataset = config.dataset
        self.loss = config.loss_type

        # Ablation variant support
        self.is_ablation_variant = getattr(config, 'is_ablation_variant', False)
        self.ablation_config = getattr(config, 'ablation_config', None)
        self.custom_loss_class = None  # Will be set during build_model for ablations

        # Report file
        self.report_file = config.report_file

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.image_size = config.image_size  # Store image size for model initialization

        self.augmentation_prob = config.augmentation_prob
        
        # Image saving control
        self.save_images = getattr(config, 'save_images', False)  # Default to False if not specified
        
   
        
        # Enhanced device detection for MPS (Apple Silicon) and CUDA
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 Using NVIDIA CUDA")
        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU (no GPU acceleration available)")
            
        print(f"Device: {self.device}")
        
        self.criterion1 = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion2 = mIoULoss(threshold=config.loss_threshold).to(self.device)
        self.criterion3 = DiceLoss(threshold=config.loss_threshold).to(self.device)

        # Model-specific parameters for VMUNet (with defaults)
        self.depths = getattr(config, 'depths', [2, 2, 9, 2])
        self.depths_decoder = getattr(config, 'depths_decoder', [2, 9, 2, 2])
        self.dims = getattr(config, 'dims', [48, 96, 192, 384])  # Compact dims for ~17M params
        self.drop_path_rate = getattr(config, 'drop_path_rate', 0.2)
        self.load_ckpt_path = getattr(config, 'load_ckpt_path', None)

        

       


    def build_model(self):
        """Build generator and discriminator."""
        print("initialize training...")

        # Check if this is an ablation variant - MUST BE FIRST
        if hasattr(self, 'is_ablation_variant') and self.is_ablation_variant:
            try:
                ablation_config = self.ablation_config
                model_class = ablation_config['model']
                self.unet = model_class(in_channels=self.img_ch, num_classes=self.output_ch, input_size=self.image_size)
                self.custom_loss_class = ablation_config['loss']
                print(f"🧪 Ablation Variant: {self.model_type}")
                print(f"   Model: {model_class.__name__}")
                print(f"   Loss: {self.custom_loss_class.__name__}")
            except Exception as e:
                print(f"Error loading ablation variant: {e}")
                raise
            
        # All other models - standard selection logic
        elif self.model_type == 'LM_Net':
            self.unet = LM_Net_PDAM(img_ch=self.img_ch, output_ch=self.output_ch, use_pdam=False)
            print("🧪 Ablation Study: LM_Net Baseline WITHOUT Pyramid Dense Attention Module (PDAM)")
            
        elif self.model_type == 'LM_Net_eem':
            self.unet = LM_Net_eem(img_ch=self.img_ch, output_ch=self.output_ch)
            print("🧪 Ablation Study: LM_Net WITHOUT Edge Enhancement Module (EEM)")
            
        elif self.model_type == 'LM_Net_PFM':
            self.unet = LM_Net_PFM(img_ch=self.img_ch, output_ch=self.output_ch, use_pfm=False)
            print("🧪 Ablation Study: LM_Net WITHOUT Parallel Feature Modules (PFM)")
            
        elif self.model_type == 'LM_Net_PDAM':
            self.unet = LM_Net_PDAM(img_ch=self.img_ch, output_ch=self.output_ch, use_pdam=True)
            print("🧪 Ablation Study: LM_Net with Pyramid Dense Attention Module (PDAM)")
            
        elif self.model_type == 'LM_Net_dw':
            self.unet = EnhancedLM_Net(img_ch=self.img_ch, output_ch=self.output_ch, use_dw_conv=False)
            print("🧪 Ablation Study: LM_Net WITHOUT Depthwise Convolutions")
            
        elif self.model_type == 'CAM_Net':
            self.unet = CAM_Net(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Original CAM_Net with all attention mechanisms")
            
        elif self.model_type == 'CAM_Net_Baseline':
            self.unet = CAM_Net_Baseline(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Ablation Study: CAM_Net Baseline WITHOUT attention mechanisms")
            
        elif self.model_type == 'CAM_Net_MSM':
            self.unet = CAM_Net_MSM(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Ablation Study: CAM_Net Baseline + MSM (Multiscale Self-attention Module)")
            
        elif self.model_type == 'CAM_Net_FEM':
            self.unet = CAM_Net_FEM(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Ablation Study: CAM_Net Baseline + FEM (Feature Extraction Module)")
            
        elif self.model_type == 'CAM_Net_ASPP':
            self.unet = CAM_Net_ASPP(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Ablation Study: CAM_Net Baseline + ASPP (Atrous Spatial Pyramid Pooling)")
            
        elif self.model_type == 'CAM_Net_EEA':
            self.unet = CAM_Net_EEA(img_in=self.img_ch, segout=self.output_ch)
            print("🧪 Ablation Study: CAM_Net Baseline + EEA (Edge Enhancement Attention)")

        # PRIMARY MODEL: DCGNet
        elif self.model_type in ['DCGNet', 'default']:
            self.unet = DCGNet(in_channels=self.img_ch, num_classes=self.output_ch, input_size=self.image_size)
            self.custom_loss_class = DCGNetLoss()
            print("DCGNet: 1.31M params | Dual-Domain Confidence-Gated Boundary Refinement")

        elif self.model_type in ['EADSNet', 'FinalEnhancedEADSNet-v3-fixed', 'FinalEnhancedEADSNet'] and EADSNet is not None:
            self.unet = EADSNet(in_channels=self.img_ch, num_classes=self.output_ch,
                                base_channels=20, target_size=225)
            print("EADSNet")

        elif self.model_type == 'EADSNetV2' and EADSNetV2 is not None:
            # EADSNet-V2: Resource-Constrained with Dual-Domain Edge Detection
            self.unet = EADSNetV2(in_channels=self.img_ch, num_classes=self.output_ch, input_size=self.image_size)
            print("🚀 USING: EADSNet-V2 (Advanced ~2M Parameters)")

        elif self.model_type == 'VMUNet':
            self.unet = VMUNet(
                input_channels=self.img_ch,
                num_classes=self.output_ch,
                depths=[2, 2, 2, 2],  # V1 original configuration
                depths_decoder=[2, 2, 2, 1],  # V1 original configuration  
                drop_path_rate=self.drop_path_rate,
                load_ckpt_path=self.load_ckpt_path
            )
            print("🧪 VMUNet V1: Vision Mamba U-Net Type 1 {2,2,2,2-2,2,2,1} - Original")
        
        elif self.model_type == 'VMUNetV2':
            # VMUNetV2 with 23.16M parameters configuration (close to target 22.77M)
            self.unet = VMUNetV2(
                input_channels=self.img_ch,
                num_classes=self.output_ch,
                mid_channel=16,  # Adjusted for 23.16M parameters (safe working config)
                dims=self.dims,  # Use config dims [16, 32, 64, 128] for ~22.77M params
                depths=[2, 2, 9, 2],  # V2 optimal configuration from Table 4
                depths_decoder=[2, 9, 2, 2],  # V2 optimal configuration
                drop_path_rate=self.drop_path_rate,
                load_ckpt_path=self.load_ckpt_path,
                deep_supervision=True  # VMUNetV2 default
            )
            if self.load_ckpt_path:
                print("🧪 VMUNet V2: Vision Mamba U-Net {2,2,9,2-2,9,2,2} with Enhanced Attention + Pretrained Weights")
            else:
                print("🧪 VMUNet V2: Vision Mamba U-Net {2,2,9,2-2,9,2,2} with Enhanced Attention (23.16M params)")

        # Shufflenetv2
        elif self.model_type =='Shufflenetv2':
            self.unet = ShuffleNetV2Seg(shufflenets[1], 232, 464, self.output_ch)
        
        # Mobilenetv1
        elif self.model_type =='Mobilenetv1':
            self.unet = MobileNetV1(num_classes=self.output_ch)
        # Mobilenetv2
        elif self.model_type == 'Mobilenetv2':
            self.unet = MobileNetV2(num_classes=self.output_ch)
           
        # Mobilenetv3
        elif self.model_type == 'Mobilenetv3':
         self.unet = MobileNetV3Seg(48, 576, self.output_ch, 'small') 

         # Mobilenetv4
        elif self.model_type =='Mobilenetv4':
            self.unet = mobilenetv4_conv_small(num_classes=self.output_ch)

        # LightM-UNet
        elif self.model_type == 'LightMUNet':
            self.unet = LightMUNet_2D(img_ch=self.img_ch, output_ch=self.output_ch, init_filters=16)
            print("🧪 LightM-UNet: Lightweight Mamba-based U-Net for medical image segmentation")

        # ULS_MSA - Ultra-Lightweight Subspace Multi-Scale Attention Network
        elif self.model_type == 'ULS_MSA':
            self.unet = ULS_MSA(img_ch=self.img_ch, output_ch=self.output_ch)
            print("🚀 ULS_MSA: Ultra-Lightweight Subspace Multi-Scale Attention Network")
            print("   Features: ULSAM, MobileViT blocks, Enhanced EEM/PFM/PDAM with DSC")

        else:
            self.unet = U_Net(self.img_ch, self.output_ch)


        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.unet.parameters(),self.lr, [self.beta1, self.beta2], weight_decay=1e-4)
        elif self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr, betas=[self.beta1, self.beta2], 
                                       weight_decay=self.weight_decay if hasattr(self, 'weight_decay') else 0.01)
        else:
            self.optimizer = optim.SGD(self.unet.parameters(), lr=self.lr, momentum=self.beta1, weight_decay=2e-4)

        # Use CosineAnnealingWarmRestarts for better convergence
        if hasattr(self, 'weight_decay') and self.optimizer_type == 'AdamW':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        else:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay)

        self.unet.to(self.device)

        # Load pretrained weights for models that have load_from method
        if hasattr(self.unet, 'load_from') and self.load_ckpt_path is not None:
            print(f"Loading pretrained weights from {self.load_ckpt_path}")
            self.unet.load_from()

        self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.report.write('\n'+str(model))
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def train(self,loss):
        factor = 0.8
        t = time.time()
        self.loss = loss
        isExist = os.path.exists(self.result_path + self.model_type+ '_' + loss)
        if not isExist:
            os.makedirs(self.result_path + self.model_type + '_' + loss)
        self.result_path_loss = os.path.join(self.result_path, self.model_type + '_' + loss) + '/' # Corrected path concatenation
        self.report = open(
            self.result_path_loss+ self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.txt',
            'a+')
        self.report.write('\n' + str(datetime.now()))

        self.f1 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_train.csv'),
                       'a', encoding='utf-8', newline='')
        self.f2 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_val.csv'),
                       'a', encoding='utf-8', newline='')
        self.model_save_path = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        self.model_save_path1 = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss)

        self.build_model()
        wr1 = csv.writer(self.f1)
        wr1.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])
        wr2 = csv.writer(self.f2)
        wr2.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])

        # U-Net Train
        if os.path.isfile(self.model_save_path):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(self.model_save_path, weights_only=False)
                print('%s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
                self.report.write('\n %s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
            except Exception as e:
                print(f"Warning: Could not load existing model from {self.model_save_path}")
                print(f"Error: {e}")
                print("Starting training from scratch...")
                self.report.write(f'\n Warning: Could not load existing model from {self.model_save_path}')
                self.report.write(f'\n Error: {e}')
                self.report.write('\n Starting training from scratch...')
                # Continue with fresh training
                best_unet_score = 0.
                results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

                for epoch in range(self.num_epochs):
                    self.unet.train(True)  # Explicitly set to training mode
                    train_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length = 0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.train_loader):
                        # print('image')
                        # print(i)
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
    # ----------------------------------UNet--------------------------------------------------------------

                        SR = self.unet(image)
                        
                        # Handle custom loss for ablation variants
                        if self.custom_loss_class is not None and self.loss == 'ablation_custom':
                            # Ablation variants with custom loss function
                            # SR is already the full output tuple from the model
                            loss_fn = self.custom_loss_class().to(self.device)
                            total_loss, loss_dict = loss_fn(SR, GT)
                            
                            # Extract main output for metrics computation
                            # Model returns (final, deep_out, edge1, edge2, edge3)
                            if isinstance(SR, tuple):
                                SR_main = SR[0]
                            else:
                                SR_main = SR
                        else:
                            # Standard loss computation for non-ablation models
                            # Handle different model output formats
                            if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4']:
                                # MobileNet models return output directly
                                SR_main = SR.contiguous().view(-1)
                            elif self.model_type in ['VMUNet', 'VMUNetV2', 'LightMUNet', 'EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed', 'ULS_MSA']:
                                # These models return single tensor output directly
                                SR_main = SR.contiguous().view(-1)
                            else:
                                # Other models (V4, V4-Lite, V3-Enhanced) return output as tuple/list
                                SR_main = SR[0]
                                SR_main = SR_main.contiguous().view(-1)
                        
                            GT_flat = GT.contiguous().view(-1)

                            loss1 = self.criterion1(SR_main, GT_flat)
                            loss2 = self.criterion2(SR_main, GT_flat)
                            loss3 = self.criterion3(SR_main, GT_flat)

                            #total_loss = loss1 + loss2 + loss3
                            total_loss = loss1 + (factor*(loss2+loss3))
                        
                        self.reset_grad()
                        total_loss.backward()
                        self.optimizer.step()

                        # For metrics, use the main output
                        if isinstance(SR_main, torch.Tensor):
                            SR_metrics = SR_main.detach()
                        else:
                            SR_metrics = SR_main[0].detach() if isinstance(SR_main, (list, tuple)) else SR_main.detach()
                        
                        GT_metrics = GT.contiguous().view(-1).detach()

                        train_loss += total_loss.detach().item()
                        acc += get_accuracy(SR_metrics, GT_metrics)
                        RC += get_Recall(SR_metrics, GT_metrics)
                        SP += get_specificity(SR_metrics, GT_metrics)
                        PC += get_Precision(SR_metrics, GT_metrics)
                        F1 += get_F1(SR_metrics, GT_metrics)
                        buff = get_mIoU(SR_metrics, GT_metrics)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR_metrics, GT_metrics)
                        length += 1

                    train_loss = train_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length

                    results[0][1].append((train_loss))
                    results[1][1].append((acc*100))
                    results[2][1].append((RC*100))
                    results[3][1].append((SP*100))
                    results[4][1].append((PC*100))
                    results[5][1].append((F1*100))
                    results[6][1].append((IoU*100))
                    results[7][1].append((mIoU*100))
                    results[8][1].append((DC*100))

                    print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    wr1.writerow(
                        [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                    writer.add_scalar("Loss/train", train_loss, epoch+1)
                    writer.add_scalar("Precision/train", PC, epoch + 1)
                    writer.add_scalar("Recall/train", RC, epoch + 1)
                    writer.add_scalar("F1 Score/train", F1, epoch + 1)
                    writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

    #===================================== Validation ====================================#
                    self.unet.train(False)
                    self.unet.eval()
                    valid_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length=0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.valid_loader):
                        
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
                        GT_original = GT  # Keep original for image saving
                        GT_f = GT

    #-------------------------------------UNet-------------------------------------------------------
                        SR = self.unet(image)
                        
                        # Handle different model output formats and custom loss
                        if self.custom_loss_class is not None and self.loss == 'ablation_custom':
                            # Ablation variants with custom loss function
                            SR_original = SR[0] if isinstance(SR, tuple) else SR
                            loss_fn = self.custom_loss_class().to(self.device)
                            total_loss, loss_dict = loss_fn(SR, GT)
                            SR_f = SR_original.contiguous().view(-1)
                        else:
                            # Standard loss computation
                            if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2', 'LightMUNet', 'EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed', 'ULS_MSA']:
                                # MobileNet, VMUNet, LightMUNet, EADSNet, EADSNetV2, and ULS_MSA models return output directly
                                SR_original = SR  # Keep original for image saving
                                SR_f = SR.contiguous().view(-1)
                            else:
                                # Other models (V4, V4-Lite, V3-Enhanced) return output as tuple/list
                                SR_original = SR[0]  # Keep original for image saving
                                SR = SR[0]
                                SR_f = SR.contiguous().view(-1)
                      
                            GT_f = GT.contiguous().view(-1)
                            loss_val_1 = self.criterion1(SR_f, GT_f)
                            loss_val_2 = self.criterion2(SR_f, GT_f)
                            loss_val_3 = self.criterion3(SR_f,GT_f)

                            #total_loss = loss_val_1 
                            total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                        # Apply sigmoid to convert logits to probabilities for metric computation
                        SR_f_prob = torch.sigmoid(SR_f.detach())
                        GT_f = GT_f.detach()

                        valid_loss += total_loss.detach().item()
                        acc += get_accuracy(SR_f_prob,GT_f)
                        RC += get_Recall(SR_f_prob,GT_f)
                        SP += get_specificity(SR_f_prob,GT_f)
                        PC += get_Precision(SR_f_prob,GT_f)
                        F1 += get_F1(SR_f_prob,GT_f)
                        buff = get_mIoU(SR_f_prob,GT_f)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR_f_prob,GT_f)
                        length += 1

                    valid_loss = valid_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length
                    unet_score = mIoU

                    results[0][2].append((valid_loss))
                    results[1][2].append((acc*100))
                    results[2][2].append((RC*100))
                    results[3][2].append((SP*100))
                    results[4][2].append((PC*100))
                    results[5][2].append((F1*100))
                    results[6][2].append((IoU*100))
                    results[7][2].append((mIoU*100))
                    results[8][2].append((DC*100))

                    print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                    wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                    writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                    writer.add_scalar("Precision/val", PC, epoch + 1)
                    writer.add_scalar("Recall/val", RC, epoch + 1)
                    writer.add_scalar("F1 Score/val", F1, epoch + 1)
                    writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                    # Handle different types of schedulers
                    if hasattr(self, 'weight_decay') and self.optimizer_type == 'AdamW':
                        self.lr_scheduler.step()  # CosineAnnealingWarmRestarts doesn't need metric
                    else:
                        self.lr_scheduler.step(valid_loss)  # ReduceLROnPlateau needs metric

                    if unet_score > best_unet_score:
                        best_unet_score = unet_score
                        print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        torch.save(self.unet,self.model_save_path)
                    epoch_custom = epoch + 1
                    if epoch_custom % 10 ==0:
                        torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                    if unet_score > 0.9:
                        torchvision.utils.save_image(image.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)
        else:
            best_unet_score = 0.
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

            for epoch in range(self.num_epochs):
                self.unet.train(True)
                train_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length = 0
                buff = []

                for i, (image, GT, name) in enumerate(self.train_loader):
                    # print('image')
                    # print(i)
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
# ----------------------------------UNet--------------------------------------------------------------

                    # Handle ablation variants first
                    if self.custom_loss_class is not None and self.loss == 'ablation_custom':
                            # Ablation variants with custom loss function
                        SR = self.unet(image)
                        loss_fn = self.custom_loss_class().to(self.device)
                        total_loss, loss_dict = loss_fn(SR, GT)
                        
                        # Extract main output for metrics
                        if isinstance(SR, tuple):
                            SR_main = SR[0].contiguous().view(-1)
                        else:
                            SR_main = SR.contiguous().view(-1)
                        GT_flat = GT.contiguous().view(-1)
                        
                    elif self.model_type == 'EnhancedEADSNetV2':
                        if self.unet.training:
                            SR, sup2, sup3 = self.unet(image)
                            # Reshape targets to match predictions
                            GT_main = F.interpolate(GT, size=SR.shape[2:], mode='nearest')
                            
                            # Calculate auxiliary losses with BCE
                            loss_main = self.criterion1(SR, GT_main)
                            loss_sup2 = self.criterion1(sup2, GT_main)
                            loss_sup3 = self.criterion1(sup3, GT_main)
                            
                            # Calculate auxiliary losses with dice
                            loss_main_dice = self.criterion2(SR, GT_main)
                            loss_sup2_dice = self.criterion2(sup2, GT_main)
                            loss_sup3_dice = self.criterion2(sup3, GT_main)
                            
                            # Combine losses with weights
                            loss = (loss_main + loss_main_dice) + 0.3 * (loss_sup2 + loss_sup2_dice) + 0.2 * (loss_sup3 + loss_sup3_dice)
                        else:
                            SR = self.unet(image)
                    elif self.model_type == 'DCGNet':
                        outputs = self.unet(image)
                        dcgnet_loss_fn = DCGNetLoss()
                        total_loss, _ = dcgnet_loss_fn(outputs, GT)
                        SR = outputs[0]
                        SR_main = SR.contiguous().view(-1)
                        GT_flat = GT.contiguous().view(-1)
                        loss = total_loss
                    elif self.model_type in ['EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed']:
                        # Handle V3-Fixed models with deep supervision
                        outputs = self.unet(image)
                        if self.model_type == 'EADSNetV2' and isinstance(outputs, tuple) and len(outputs) == 5:
                            # EADSNetV2 returns tuple: (mask, sup2, sup3, edge2, edge3)
                            SR, sup2, sup3, edge2, edge3 = outputs
                            
                            # Calculate main loss
                            SR_flat = SR.contiguous().view(-1)
                            GT_flat = GT.contiguous().view(-1)
                            
                            loss_main_bce = self.criterion1(SR_flat, GT_flat)
                            loss_main_dice = self.criterion3(SR_flat, GT_flat)
                            
                            # Calculate auxiliary losses with same GT
                            sup2_flat = sup2.contiguous().view(-1)
                            sup3_flat = sup3.contiguous().view(-1)
                            
                            loss_sup2_bce = self.criterion1(sup2_flat, GT_flat)
                            loss_sup2_dice = self.criterion3(sup2_flat, GT_flat)
                            
                            loss_sup3_bce = self.criterion1(sup3_flat, GT_flat)
                            loss_sup3_dice = self.criterion3(sup3_flat, GT_flat)
                            
                            # Total loss with weighted auxiliary losses
                            total_loss = (loss_main_bce + loss_main_dice) + \
                                       0.4 * (loss_sup2_bce + loss_sup2_dice) + \
                                       0.2 * (loss_sup3_bce + loss_sup3_dice)
                        elif isinstance(outputs, tuple) and len(outputs) == 3:
                            # EADSNet/V3-Fixed returns (main, aux3, aux2)
                            SR, aux3, aux2 = outputs
                            
                            # Calculate main loss
                            SR_flat = SR.contiguous().view(-1)
                            GT_flat = GT.contiguous().view(-1)
                            
                            loss_main_bce = self.criterion1(SR_flat, GT_flat)
                            loss_main_dice = self.criterion3(SR_flat, GT_flat)
                            
                            # Calculate auxiliary losses with same GT
                            aux3_flat = aux3.contiguous().view(-1)
                            aux2_flat = aux2.contiguous().view(-1)
                            
                            loss_aux3_bce = self.criterion1(aux3_flat, GT_flat)
                            loss_aux3_dice = self.criterion3(aux3_flat, GT_flat)
                            
                            loss_aux2_bce = self.criterion1(aux2_flat, GT_flat)
                            loss_aux2_dice = self.criterion3(aux2_flat, GT_flat)
                            
                            # Total loss with weighted auxiliary losses
                            total_loss = (loss_main_bce + loss_main_dice) + \
                                       0.4 * (loss_aux3_bce + loss_aux3_dice) + \
                                       0.2 * (loss_aux2_bce + loss_aux2_dice)
                        else:
                            # Inference mode
                            SR = outputs
                            SR_flat = SR.contiguous().view(-1)
                            GT_flat = GT.contiguous().view(-1)
                            
                            # Standard loss calculation
                            loss1 = self.criterion1(SR_flat, GT_flat)
                            loss2 = self.criterion2(SR_flat, GT_flat)
                            loss3 = self.criterion3(SR_flat, GT_flat)
                            total_loss = loss1 + (factor*(loss2+loss3))
                    else:
                        # Handle standard models (U-Net, MobileNet, etc.)
                        SR = self.unet(image)
                        if isinstance(SR, tuple):
                            SR = SR[0]  # Extract main output if tuple

                    
                    # Handle different model output formats for loss calculation
                    if self.model_type == 'DCGNet':
                        SR_for_loss = SR_main
                        GT_for_loss = GT_flat
                    elif self.model_type in ['EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed']:
                        # V3-Fixed/V2 models already handled above
                        # Use the already computed SR_flat and GT_flat for metrics
                        SR_for_loss = SR_flat
                        GT_for_loss = GT_flat
                    elif self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet', 'EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed', 'ULS_MSA']:
                        # Models that return output directly
                        if not isinstance(outputs, tuple):
                            SR_flat = SR.contiguous().view(-1)
                            GT_flat = GT.contiguous().view(-1)
                        else:
                            # For deep supervision models during inference, use main output
                            SR_flat = SR.contiguous().view(-1)
                            GT_flat = GT.contiguous().view(-1)
                        SR_for_loss = SR_flat
                        GT_for_loss = GT_flat
                    else:
                        # Other models return output as tuple/list
                        SR = SR[0]
                        SR_flat = SR.contiguous().view(-1)
                        GT_flat = GT.contiguous().view(-1)
                        SR_for_loss = SR_flat
                        GT_for_loss = GT_flat
                    
                    # Calculate loss if not already done by deep supervision or ablation variants
                    if self.model_type not in ['EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed'] and not (self.custom_loss_class is not None and self.loss == 'ablation_custom'):
                        # Standard loss calculation for non-V3-Fixed models and non-ablation models
                        loss1 = self.criterion1(SR_for_loss, GT_for_loss)
                        loss2 = self.criterion2(SR_for_loss, GT_for_loss)
                        loss3 = self.criterion3(SR_for_loss, GT_for_loss)
                        total_loss = loss1 + (factor*(loss2+loss3))

                    self.reset_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        if self.custom_loss_class is not None and self.loss == 'ablation_custom':
                            # For ablation variants, use SR_main that was set earlier
                            SR_prob_flat = torch.sigmoid(SR_main.detach())
                            GT_metric = GT_flat.detach()
                        elif self.model_type in ['EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed']:
                            # For V3-Fixed model, use the flattened main output
                            SR_prob_flat = torch.sigmoid(SR_flat.detach())
                            GT_metric = GT_flat.detach()
                        else:
                            SR_prob_flat = torch.sigmoid(SR_for_loss.detach())
                            GT_metric = GT_for_loss.detach()

                    train_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                train_loss = train_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length

                results[0][1].append((train_loss))
                results[1][1].append((acc*100))
                results[2][1].append((RC*100))
                results[3][1].append((SP*100))
                results[4][1].append((PC*100))
                results[5][1].append((F1*100))
                results[6][1].append((IoU*100))
                results[7][1].append((mIoU*100))
                results[8][1].append((DC*100))

                print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                wr1.writerow(
                    [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                writer.add_scalar("Loss/train", train_loss, epoch+1)
                writer.add_scalar("Precision/train", PC, epoch + 1)
                writer.add_scalar("Recall/train", RC, epoch + 1)
                writer.add_scalar("F1 Score/train", F1, epoch + 1)
                writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

#===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                valid_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length=0
                buff = []

                for i, (image, GT, name) in enumerate(self.valid_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    GT_original = GT  # Keep original for image saving
                    GT_f = GT

#-------------------------------------UNet-------------------------------------------------------
                    SR = self.unet(image)
                    
                    # Handle ablation variants in eval mode - they return single tensor
                    if self.custom_loss_class is not None and self.loss == 'ablation_custom':
                        # In eval mode, ablation variants return just the final output
                        # SR is already a single tensor
                        SR_original = SR
                        SR_f = SR.contiguous().view(-1)
                        GT_f = GT.contiguous().view(-1)
                        
                        # Calculate standard loss for validation reporting
                        loss_val_1 = self.criterion1(SR_f, GT_f)
                        loss_val_2 = self.criterion2(SR_f, GT_f)
                        loss_val_3 = self.criterion3(SR_f, GT_f)
                    # Handle different model output formats
                    elif self.model_type == 'DCGNet':
                        # DCGNet returns single tensor in eval mode
                        SR_original = SR
                        SR_f = SR.contiguous().view(-1)
                        GT_f = GT.contiguous().view(-1)
                    elif self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet', 'EADSNet', 'EADSNetV2', 'FinalEnhancedEADSNet-v3-fixed', 'ULS_MSA']:
                        # Handle models that may return tuples during training
                        if isinstance(SR, tuple):
                            SR = SR[0]  # Use main output for validation
                        SR_original = SR  # Keep original for image saving
                        SR_f = SR.contiguous().view(-1)
                        GT_f = GT.contiguous().view(-1)
                    else:
                        # Other models return output as tuple/list
                        SR_original = SR[0]  # Keep original for image saving
                        SR = SR[0]
                        SR_f = SR.contiguous().view(-1)
                        GT_f = GT.contiguous().view(-1)
                   
                    # Calculate loss if not already done by ablation branch
                    if not (self.custom_loss_class is not None and self.loss == 'ablation_custom'):
                        loss_val_1 = self.criterion1(SR_f, GT_f)
                        loss_val_2 = self.criterion2(SR_f, GT_f)
                        loss_val_3 = self.criterion3(SR_f, GT_f)

                    total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        SR_prob_flat = torch.sigmoid(SR_f.detach())
                        GT_metric = GT_f.detach()

                    valid_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                valid_loss = valid_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length
                unet_score = mIoU

                results[0][2].append((valid_loss))
                results[1][2].append((acc*100))
                results[2][2].append((RC*100))
                results[3][2].append((SP*100))
                results[4][2].append((PC*100))
                results[5][2].append((F1*100))
                results[6][2].append((IoU*100))
                results[7][2].append((mIoU*100))
                results[8][2].append((DC*100))

                print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                writer.add_scalar("Precision/val", PC, epoch + 1)
                writer.add_scalar("Recall/val", RC, epoch + 1)
                writer.add_scalar("F1 Score/val", F1, epoch + 1)
                writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                # Handle different types of schedulers
                if hasattr(self, 'weight_decay') and self.optimizer_type == 'AdamW':
                    self.lr_scheduler.step()  # CosineAnnealingWarmRestarts doesn't need metric
                else:
                    self.lr_scheduler.step(valid_loss)  # ReduceLROnPlateau needs metric

                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(self.unet,self.model_save_path)
                epoch_custom = epoch + 1
                if epoch_custom % 10 ==0:
                    torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                if unet_score > 0.9:
                    torchvision.utils.save_image(image.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                
                    torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                    
                    torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

              

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)

        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        self.f1.close()
        self.f2.close()
        writer.close()

    def get_gradCAM(self,image,SR, GT,size):
        total_loss = self.criterion1(SR, GT)
        total_loss.backward()
        gradients = self.unet.get_activation_gradients()
        pooled_gradients = torch.mean(gradients, dim=[0,2,3])
        activations = self.unet.get_activations(image).detach()
        for i in range(activations.shape[1]):
            activations[:,i,:,:] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
        heatmap = nn.ReLU()(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        image = image.squeeze(0)
        image = image.permute(1,2,0)
        image = image.cpu().numpy()
        image = np.uint8(image * 255)
        heatmap = cv2.resize(heatmap, (320, 320))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        return overlay, heatmap

    def test(self, loss, data, model): 
        # Construct model path with proper path joining
        model_file_path = os.path.join(self.model_path, self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        
        # If Test_ prefixed model not found, try Train_ prefixed (for ablations trained without Test prefix)
        if not os.path.isfile(model_file_path) and self.report_file.startswith('Test_'):
            fallback_report = self.report_file.replace('Test_', 'Train_', 1)
            model_file_path = os.path.join(self.model_path, fallback_report + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        
        if os.path.isfile(model_file_path):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(model_file_path, weights_only=False)
                print('%s is Successfully Loaded from %s' % (self.model_type, model_file_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Trained model NOT found or could not be loaded for {self.model_type} with loss {loss}, Please train a model first")
                return
        else:
            print(f"Trained model NOT found for {self.model_type} with loss {loss}")
            print(f"Expected path: {model_file_path}")
            print(f"Please train a model first")
            return

        # Create SR directory structure as in original design
        isExist = os.path.exists(self.SR_path + self.model_type + '_' + loss)
        if not isExist:
            os.makedirs(self.SR_path + self.model_type + '_' + loss)
        self.model_path_loss = self.SR_path + self.model_type + '_' + loss + '/'
        self.test_acc = open(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv', 'a+')

        wr_test = csv.writer(self.test_acc)
        if os.path.getsize(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv') == 0:
            wr_test.writerow(['Model', 'SE (%)', 'SP (%)', 'ACC (%)', 'IoU (%)', 'Dice (%)', 'Params(M)', 'FLOPs', 'Avg Inference Time'])


        self.unet.train(False)
        self.unet.eval()

        input_size_flops = (self.img_ch, 224, 224)
        try:
            macs, params = get_model_complexity_info(self.unet, input_size_flops, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
            flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
            flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
        except Exception as e:
            print(f"Could not calculate FLOPs/Params for {self.model_type}: {e}")
            macs, params, flops, flops_unit = "N/A", "N/A", "N/A", ""

        print(f'Computational complexity: {macs}')
        print(f'Computational complexity: {flops} {flops_unit}Flops')
        print(f'Number of parameters: {params}')

        acc = 0.
        RC = 0.
        SP = 0.
        PC = 0.
        F1 = 0.
        IoU = 0
        mIoU = 0.
        DC = 0.
        total_elapsed = 0.
        length = 0

        with torch.no_grad():
            for i, (image, GT, name) in enumerate(self.test_loader):
                image = image.to(self.device)
                GT = GT.to(self.device)

                start_time = time.time()
                SR = self.unet(image)
                elapsed = time.time() - start_time
                total_elapsed += elapsed
                
                # Handle different model output formats
                if self.model_type in ['DCGNet', 'Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet', 'EADSNet', 'FinalEnhancedEADSNet-v3-fixed', 'ULS_MSA', 'EADSNetV2']:
                    # Handle models that may return tuples during inference
                    if isinstance(SR, tuple):
                        SR = SR[0]  # Use main output for testing
                    SR_f = SR.contiguous().view(-1)
                    SR_f_sigmoid = torch.sigmoid(SR_f)
                    SR_sigmoid = torch.sigmoid(SR)
                else:
                    # Other models return output as tuple/list
                    SR = SR[0]  # Modify this based on your model's output structure
                    SR_f = SR.contiguous().view(-1)
                    SR_f_sigmoid = torch.sigmoid(SR_f)
                    SR_sigmoid = torch.sigmoid(SR)
                GT_f = GT.contiguous().view(-1)

                acc += get_accuracy(SR_f_sigmoid, GT_f)
                RC += get_Recall(SR_f_sigmoid, GT_f)
                SP += get_specificity(SR_f_sigmoid, GT_f)
                PC += get_Precision(SR_f_sigmoid, GT_f)
                F1 += get_F1(SR_f_sigmoid, GT_f)
                buff = get_mIoU(SR_f_sigmoid, GT_f)
                IoU += buff[0]
                mIoU += buff[1]
                DC += get_DC(SR_f_sigmoid, GT_f)
                length += 1

                threshold = 0.5
                
                # Original processing for other models
                SR_processed = torch.sigmoid(SR).squeeze(1)
                SR_processed[SR_processed < threshold] = 0
                SR_processed[SR_processed >= threshold] = 1

                # Only save images if save_images flag is True
                if self.save_images:
                    for j in range(SR_processed.shape[0]):
                        im = Image.fromarray(SR_processed[j].cpu().numpy() * 255).convert('L')
                        imo = im.resize((256, 256), resample=Image.BILINEAR)
                        imo.save(self.model_path_loss + name[j])

        acc /= length
        RC /= length
        SP /= length
        PC /= length
        F1 /= length
        IoU /= length
        mIoU /= length
        DC /= length

        # Convert to percentages and format properly
        SE_percent = RC * 100  # Sensitivity (Recall)
        SP_percent = SP * 100  # Specificity
        ACC_percent = acc * 100  # Accuracy
        IoU_percent = IoU * 100  # IoU
        Dice_percent = DC * 100  # Dice
        
        # Extract numeric values for params and FLOPs
        try:
            params_numeric = float(re.findall(r'([\d.]+)', params)[0])
        except:
            params_numeric = 0.0
            
        try:
            flops_numeric = flops
            flops_unit_str = flops_unit + "Flops"
        except:
            flops_numeric = 0.0
            flops_unit_str = "GFlops"

        total_images_processed = length * self.test_loader.batch_size
        avg_inference_time = total_elapsed / total_images_processed if total_images_processed > 0 else 0

        wr_test.writerow([self.model_type, f"{SE_percent:.2f}", f"{SP_percent:.2f}", f"{ACC_percent:.2f}", 
                         f"{IoU_percent:.2f}", f"{Dice_percent:.2f}", f"{params_numeric:.2f}", 
                         f"{flops_numeric:.2f}", f"{avg_inference_time:.6f}"])
        print('Results have been Saved')
        print(f'Average Inference Time per Image: {avg_inference_time:.6f} seconds')

        self.test_acc.close()
        
        # Save LaTeX formatted metrics
        latex_file_path = self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_latex.txt'
        with open(latex_file_path, 'w') as latex_file:
            # Write header row
            latex_file.write(r"\textbf{Model} & \textbf{RC} & \textbf{PR} & \textbf{F1} & \textbf{mIoU} & \textbf{Params} \\" + "\n")
            # Write data row
            latex_file.write(f"{self.model_type} & {RC*100:.2f} & {PC*100:.2f} & {F1*100:.2f} & {mIoU*100:.2f} & {params_numeric:.2f} \\\\\n")
        
        print(f"LaTeX metrics saved to: {latex_file_path}")
        
        # Save CSV formatted metrics
        csv_file_path = self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_metrics.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Model', 'RC (%)', 'PR (%)', 'F1 (%)', 'mIoU (%)', 'Params (M)'])
            csv_writer.writerow([self.model_type, f"{RC*100:.2f}", f"{PC*100:.2f}", f"{F1*100:.2f}", f"{mIoU*100:.2f}", f"{params_numeric:.2f}"])
        
        print(f"CSV metrics saved to: {csv_file_path}")