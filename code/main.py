import argparse
import os
import sys
from solver import Solver  # Training engine
from data_loader import get_loader
from torch.backends import cudnn
import random

# Add ablation_study directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ablation_study'))

# Import ablation variants registry
try:
    from ablation_variants import ABLATION_REGISTRY
    ABLATION_AVAILABLE = True
except ImportError:
    ABLATION_REGISTRY = {}
    ABLATION_AVAILABLE = False
    print("Warning: ablation_variants module not found. Ablation study features disabled.")

def main(config):
    cudnn.benchmark = True
    
    # Set report file based on mode and models if not explicitly set
    if config.report_file == 'Report':
        config.report_file = f"{config.mode.capitalize()}_{'_'.join([m.split('_')[0] for m in config.models.split(',')]) if config.models else 'AllModels'}"

    # Create model and result directories if they don't exist
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.result_path, exist_ok=True)

    print(config)

    # Load data
    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)

    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)

    # Build list of available models (standard + ablation variants)
    all_standard_models = ['DCGNet']
    all_available_models = all_standard_models + list(ABLATION_REGISTRY.keys()) if ABLATION_AVAILABLE else all_standard_models
    
    # Filter models based on command line argument
    if config.models:
        requested_models = [model.strip() for model in config.models.split(',')]
        ablation_models = [m for m in requested_models if m in all_available_models]
        if not ablation_models:
            print(f"Warning: No valid models found in '{config.models}'.")
            print(f"Available models: {all_available_models}")
            ablation_models = all_standard_models
    else:
        ablation_models = all_standard_models  # Run standard models by default

    print(f"Running models: {ablation_models}")

    # Process each model
    for model_name in ablation_models:
        print(f"\n{'='*50}")
        print(f"Processing : {model_name}")
        print(f"{'='*50}")

        # Check if this is an ablation variant
        is_ablation = model_name in ABLATION_REGISTRY if ABLATION_AVAILABLE else False
        
        # Pass ablation info to Solver
        config.is_ablation_variant = is_ablation
        if is_ablation:
            config.ablation_config = ABLATION_REGISTRY[model_name]
            
        solver = Solver(config, model_name, train_loader, valid_loader, test_loader)

        if config.mode == 'train':
            # Training phase - use variant-specific loss if ablation, else use default
            if is_ablation:
                solver.train(loss='ablation_custom')
            else:
                solver.train(loss='BCE_Dice_mIoU')

        elif config.mode == 'test':
            # Testing phase
            print(f"\n{'='*50}")
            print(f"Testing : {model_name}")
            print(f"{'='*50}")
            # Use correct loss type: ablation_custom for ablations, BCE_Dice_mIoU for standard models
            loss_type = 'ablation_custom' if is_ablation else 'BCE_Dice_mIoU'
            solver.test(loss=loss_type, data=config.dataset, model=model_name)

        else:
            print(f"Invalid mode: {config.mode}. Please use 'train' or 'test'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224)     # Standard size for better results
    parser.add_argument('--num_workers', type=int, default=0)

    # Training hyper-parameters - Research-Proven for Medical Segmentation
    parser.add_argument('--lr', type=float, default=0.001)        # Higher LR for large datasets (research-proven)
    parser.add_argument('--num_epochs', type=int, default=100)     # Standard for medical segmentation
    parser.add_argument('--num_epochs_decay', type=int, default=10) # T_0 for CosineAnnealingWarmRestarts
    parser.add_argument('--batch_size', type=int, default=4)       # Larger batch for stable gradients
    parser.add_argument('--loss_threshold', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='BCE_Dice_mIoU', help='[BCE,BCE_mIoU,BCE_Dice_mIoU]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='[Adam,SGD,AdamW]') # Standard Adam (research-proven)
    parser.add_argument('--beta1', type=float, default=0.9)        # Standard Adam momentum
    parser.add_argument('--beta2', type=float, default=0.999)      # Standard Adam momentum
    parser.add_argument('--weight_decay', type=float, default=0.00005) # Research-proven weight decay (Ma and Yarats, 2021)
    parser.add_argument('--augmentation_prob', type=float, default=0.9) # Strong augmentation for large datasets



    # Misc  
    parser.add_argument('--report_file', type=str, default='Report')
    parser.add_argument('--mode', type=str, default='train', help='[train,test]')
    parser.add_argument('--dataset', type=str, default='PH2', help='[PH2,ISIC2017,ISIC2018]')
    parser.add_argument('--use_enhanced_lmnet', action='store_true')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to run')
    parser.add_argument('--save_images', action='store_true', help='Save predicted images during testing')

   
    parser.add_argument('--train_path', type=str,  default=None)
    parser.add_argument('--valid_path', type=str,  default=None)
    parser.add_argument('--test_path', type=str,   default=None)
    parser.add_argument('--model_path', type=str,  default='/Users/razan/Documents/Research/2.Technical/1.Seg-papers/3.EADSNet/main-code/Results/Ablation/Models/')
    parser.add_argument('--result_path', type=str, default='/Users/razan/Documents/Research/2.Technical/1.Seg-papers/3.EADSNet/main-code/Results/Ablation/Results/')
    parser.add_argument('--SR_path', type=str,     default='/Users/razan/Documents/Research/2.Technical/1.Seg-papers/3.EADSNet/main-code/Results/Ablation/SR/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    
    # Set dataset paths dynamically based on dataset selection
    dataset_base_path = '/Users/razan/Documents/Research/2.Technical/0.Datasets'
    
    if config.dataset == 'PH2':
        dataset_path = f'{dataset_base_path}/PH2'
    elif config.dataset == 'ISIC2017':
        dataset_path = f'{dataset_base_path}/ISIC2017'
    elif config.dataset == 'ISIC2018':
        dataset_path = f'{dataset_base_path}/ISIC2018'
    else:
        dataset_path = f'{dataset_base_path}/{config.dataset}'
    
    # Set paths if not explicitly provided
    if config.train_path is None:
        config.train_path = f'{dataset_path}/train/'
    if config.valid_path is None:
        config.valid_path = f'{dataset_path}/valid/'
    if config.test_path is None:
        config.test_path = f'{dataset_path}/test/'
    
    # Auto-enable save_images for test mode
    if config.mode == 'test' and not config.save_images:
        config.save_images = True
    
    main(config)