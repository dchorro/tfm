#!/usr/bin/env python3

# Standard packages
from transformers import SegformerForSemanticSegmentation
import sys, os
# from SegFormer.mmseg.models.builder import build_segmentor
import matplotlib.pyplot as plt
from unet import UNet, UNet2
from attention_unet_implementation import R2AttU_Net, AttU_Net, R2U_Net
# import importlib
# importlib.reload()
from torch.utils.data import RandomSampler
from test import test_func
from schedulefree import AdamWScheduleFree
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.nn as nn
import sys, os, argparse
from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import numpy as np
from torchvision.utils import save_image, make_grid
from skimage.filters import threshold_otsu
from multiprocessing import cpu_count
from termcolor import colored
import time
from datetime import datetime
# Local packages
import procImg
from model_0 import HTRModel_0, UNet, HTRModel_ResNet
#from model_ARU import ConvNet,UNet
from dataset_p2xml import Dataset, ctc_collate
from torchvision import transforms as T
# import torchvision.transforms.v2 as v2
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context


def binarize_target(target, threshold=0.05):
    """
    Convert target tensor into a binary mask where lines are represented by 1
    and the rest is 0.

    Args:
        target (torch.Tensor): The target tensor (batch, channels, height, width).
        threshold (float): The threshold value to separate lines from the background.

    Returns:
        torch.Tensor: Binarized target tensor (same shape as input) with 0 and 1 values.
    """
    # Apply the threshold to create a binary mask
    binary_target = (target > threshold).float()
    
    return binary_target


def calculate_iou_2(pred, target):
    
    pred_ = pred.detach().clone() > 0.5  # Convert to binary mask
    # save_predictions(x=None, y=target, outputs=pred, epoch=0, save_dir="testfolder")
    target_ = target.detach().clone() > 0.5
    
    # print(target_.min(), target_.max())
    # print(pred_.min(), pred_.max())
    # save_predictions(x=None, y=target_, outputs=pred_, epoch=3, save_dir="testfolder")

    intersection = (pred_ & target_).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred_ | target_).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + 1e-6) / (union + 1e-6)
    return

def calculate_iou(pred, target):
    # save_predictions(x=None, y=target, outputs=pred, epoch=0, save_dir="testfolder")
    pred_ = pred.detach().clone() > 0.5  # Convert to binary mask
    target_ = target.detach().clone() > 0.5
    intersection = (pred_ & target_).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (pred_ | target_).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def calculate_f1(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    tp = (pred & target).float().sum((1, 2))
    fp = (pred & ~target).float().sum((1, 2))
    fn = (~pred & target).float().sum((1, 2))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.mean()


def log_hyperparameters(logs, args, model, optimizer, scheduler, criterion):
    logs.write("Experiment Hyperparameters:\n")
    logs.write(f"Architecture: {args.arch_name}\n")
    logs.write(f"Encoder: {args.encoder_name}\n")
    logs.write(f"Encoder weights: {args.encoder_weights}\n")
    logs.write(f"Dataset: {args.dataset_train}\n")
    logs.write(f"Epochs: {args.epochs}\n")
    logs.write(f"Batch size (train): {args.batch_size}\n")
    logs.write(f"Batch size (val): {args.val_batch_size}\n")
    logs.write(f"Learning rate: {args.lr}\n")
    logs.write(f"Frozen encoder: {args.frozen}\n")
    if args.frozen:
        logs.write(f"Unfreeze at epoch: {args.unfreeze}\n")
    # logs.write(f"Data augmentation: {args.data_augm}\n")
    # logs.write(f"Fixed width: {args.fixed_width}\n")
    logs.write(f"Early stopping: {args.early_stop if args.early_stop > 0 else 'Disabled'}\n")
    logs.write(f"Pretrained model: {args.base_model if args.base_model else 'None'}\n")
    
    # Log model summary
    logs.write("\nModel Summary:\n")
    logs.write(str(model))
    
    # Log optimizer
    logs.write(f"\nOptimizer: {type(optimizer).__name__}\n")
    logs.write(f"Optimizer params: {optimizer.defaults}\n")
    
    # Log scheduler
    logs.write(f"\nScheduler: {type(scheduler).__name__}\n")
    logs.write(f"Scheduler params: {scheduler.state_dict()}\n")
    
    # Log loss function
    logs.write(f"\nLoss Function: {type(criterion).__name__}\n")
    
    logs.write("\n--- Training Start ---\n\n")
    logs.flush()

def log_data_aug(logs, train_xy_transforms, train_x_transforms, val_xy_transforms):
    logs.write(f"train_xy_transforms: {train_xy_transforms}\n")
    logs.write(f"\ntrain_x_transforms: {train_x_transforms}\n")
    logs.write(f"\nval_xy_transforms: {val_xy_transforms}\n\n\n")
    logs.flush()


def downsample_proportionally_if_needed(tensor, max_height, max_width):
    _, _, height, width = tensor.size()  # Assuming tensor shape is (N, C, H, W)

    if height > max_height or width > max_width:
        # Calculate the scaling factor while maintaining aspect ratio
        scaling_factor = min(max_height / height, max_width / width)
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        # Downsample the tensor
        tensor = nnf.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False, antialias=True)
        
    return tensor


# Integration in your data preprocessing or data loading pipeline
def process_batch(x, y, max_height, max_width):
    # x = downsample_if_needed(x, max_height, max_width)
    # y = downsample_if_needed(y, max_height, max_width)


    x = downsample_proportionally_if_needed(x, max_height, max_width)
    y = downsample_proportionally_if_needed(y, max_height, max_width)
    return x, y


def buildModel(arch_name, encoder_name, encoder_weights, in_channels=1, n_classes=1):
    
    if arch_name == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            decoder_attention_type="scse"
        )
    elif arch_name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            decoder_attention_type="scse"
        )
    elif arch_name == "segformer":
        # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        return model
    

    elif arch_name == "deeplabv3p":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )

    elif arch_name == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass
    
    elif arch_name == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass

    elif arch_name == "deeplabv3":
        return smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass
    elif arch_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass
    
    elif arch_name == "linknet":
        return smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass
    
    elif arch_name == "manet":
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass
    
    elif arch_name == "pan":
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
        pass


def create_experiment_folder(base_dir, arch_name, encoder_name, encoder_weights, dataset_name, frozen, lr, unfreeze, pretrained):
    # Create a folder name based on current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    if pretrained:
        if frozen:
            folder_name = f"{current_time}_{pretrained[2:-4].replace('_', '-')}_{dataset_name}_frozen_{lr}_{unfreeze}"
        else:
            folder_name = f"{current_time}_{pretrained[2:-4].replace('_', '-')}_{dataset_name}_{lr}"
    else:
        if frozen:
            folder_name = f"{current_time}_{arch_name}_{encoder_name.replace('-','_')}_{encoder_weights}_{dataset_name}_frozen_attention_{lr}_{unfreeze}"
        else:
            folder_name = f"{current_time}_{arch_name}_{encoder_name.replace('-','_')}_{encoder_weights}_{dataset_name}_attention_{lr}"

    # Path to experiments directory
    experiment_dir = os.path.join(base_dir, folder_name)

    # Create the directory
    try:
        os.makedirs(experiment_dir)
        print(f"Successfully created experiment directory: {experiment_dir}")
    except OSError as e:
        print(f"Error creating directory {experiment_dir}: {e}")

    return experiment_dir


def pad_to_32(img):
    _, _, h, w = img.size()
    
    # Calculate the padding needed to make height and width divisible by 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    # Padding values: (padding_left, padding_right, padding_top, padding_bottom)
    padding = (0, pad_w, 0, pad_h)
    
    # Apply padding
    padded_img = F.pad(img, padding, mode='constant', value=0)
    
    return padded_img, padding

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


def BCELoss_class_weighted22(weights):
    def loss(output, target):
        output = torch.clamp(output,min=1e-7,max=1-1e-7) # assegura que la hipotesis estiga entre  [0:1] 
        bce = - weights[1] * target * torch.log(output) - weights[0] * (1 - target) * torch.log(1 - output)
        return torch.mean(bce)

    return loss

def BCELoss_class_weighted():
    def loss(output, target, weights):
        output = torch.clamp(output,min=1e-7,max=1-1e-7) # assegura que la hipotesis estiga entre  [0:1] 
        bce = - (1-weights) * target * torch.log(output) - weights * (1 - target) * torch.log(1 - output)
        return torch.mean(bce)

    return loss

class Loss_iou(torch.nn.Module):
    def __init__(self):
        super(Loss_iou, self).__init__()
  
    def forward(self, output, target):
        output = torch.sigmoid(output)
        output[output >= 0.2]=1
        output[output <  0.2]=0
        
        output_ = output.type(torch.int64)
        target_ = target.type(torch.int64)

        union = torch.sum(output_ | target_)
        intersection = torch.sum(output_ & target_)
        
      #  print("union =" +str(union.item()))
      #  print("inter =" +str(intersection.item()))
        return 1 - torch.mean(intersection.float() / union.float())


def get_loss(loss="weightbce"):
    if loss=="weightbce":
        return BCELoss_class_weighted()
    elif loss == "dice":
        return DiceLoss()


def save_predictions(x=None, y=None, outputs=None, epoch=None, save_dir="."):
    """
    Save the model's predictions, input images, and ground truth labels as image files.

    Parameters:
    - x: Input tensor (batch of images).
    - y: Ground truth labels (segmentation maps).
    - outputs: Model's predicted outputs (segmentation maps).
    - epoch: Counter to ensure unique file names for saved images.
    - save_dir: Directory where the images will be saved (default is current directory).
    """
    # Ensure directory exists (if necessary)
    os.makedirs(save_dir, exist_ok=True)

    # Process and save the predictions (thresholded at 0.5)
    if outputs != None:
        out_ = (outputs > 0.5).float()  # Convert to binary predictions
        outputs_ = make_grid(out_, nrow=2, normalize=True, scale_each=True)
        save_image(outputs_, os.path.join(save_dir, f"outputs_{epoch}.png"), nrow=1, normalize=True)

    # Process and save the input images
    if x != None:
        x_ = make_grid(x, nrow=2, normalize=True, scale_each=True)
        save_image(x_, os.path.join(save_dir, f"out_x_{epoch}.png"), nrow=1)

    if y != None:
        # Resize and save the ground truth labels
        y_ = (y > 0.5).float()  # Convert to binary predictions
        y_ = nnf.interpolate(y, size=(y.size(2), int(y.size(3))), mode='bicubic', align_corners=False)
        y_ = make_grid(y_, nrow=2, normalize=True, scale_each=True)
        save_image(y_, os.path.join(save_dir, f"out_y_{epoch}.png"), nrow=1)


    return


def train_step(model, train_loader, device, criterion, optimizer, verbosity, epoch):
    model.train()
    total_train_loss = 0
    epoch_iou = 0
    epoch_f1 = 0
    ignored_batches = []
    batch_num = 0
    threshold = 0.04
    
    try:
        terminal_cols = os.get_terminal_size()[0]
    except IOError:
        terminal_cols = 80

    format = '{l_bar}{bar:' + str(terminal_cols - 48) + '}{r_bar}'
    
    for (x, y, bIdxs) in tqdm(train_loader, bar_format=format, colour='green', desc='  Train'):
        batch_num += 1
        torch.cuda.empty_cache()

        x, y = x.to(device), y.to(device)
        
        x, _ = pad_to_32(x)
        y, _ = pad_to_32(y)
        # ----------
        # y = (y > threshold).float()
        # print(f"{x.shape=}    {y.shape=}")


        outputs = model(x)
        
        # if verbosity and batch_num == 1:
        if True and batch_num == 1:
            save_predictions(x, y, outputs, epoch, ".")
            # print("Prediction saved!")
            
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        epoch_iou += calculate_iou(outputs, y).item()
        epoch_f1 += calculate_f1(outputs, y).item()

    num_img_processed = len(train_loader) - len(ignored_batches)
    return total_train_loss / num_img_processed, epoch_iou / num_img_processed, epoch_f1 / num_img_processed, ignored_batches


def val_step(model, val_loader, device, criterion, experiment_dir, epoch):
    # model.eval()
    # model.train()
    total_val_loss = 0
    val_epoch_iou = 0
    val_epoch_f1 = 0
    first = True
    threshold = 0.04
    
    try:
        terminal_cols = os.get_terminal_size()[0]
    except IOError:
        terminal_cols = 80

    format = '{l_bar}{bar:' + str(terminal_cols - 48) + '}{r_bar}'
    torch.cuda.empty_cache()
    with torch.no_grad():
        for (x, y, bIdxs) in tqdm(val_loader, bar_format=format, colour='magenta', desc='  Train'):
            torch.cuda.empty_cache()

            x, y = x.to(device), y.to(device)
            x, _ = pad_to_32(x)
            y, _ = pad_to_32(y)
            # ----------
            # y = (y > threshold).float()
            # print(f"{x.shape=}    {y.shape=}")


            outputs = model(x)

            if first:
                save_predictions(x, y, outputs, epoch, experiment_dir)
                first = False
                
            loss = criterion(outputs, y)

            total_val_loss += loss.item()
            val_epoch_iou += calculate_iou(outputs, y).item()
            val_epoch_f1 += calculate_f1(outputs, y).item()

    return total_val_loss / len(val_loader), val_epoch_iou / len(val_loader), val_epoch_f1 / len(val_loader),

def train(model, htr_dataset_train, htr_dataset_val, logs, device, train_batch_size, val_batch_size, experiment_dir, frozen, epochs=20, lr=3e-4, 
          unfreeze=50, early_stop=10, lh=64, verbosity=False):
    # Control reproducibility
    # torch.manual_seed(17)
    
    nwrks = int(cpu_count() * 0.70)  # Use ~2/3 of available cores
    train_dataset_size = len(htr_dataset_train)
    train_indices = list(range(train_dataset_size))
    np.random.shuffle(train_indices)
    # train_loader = torch.utils.data.DataLoader(htr_dataset_train,
    #                                            batch_size=train_batch_size,
    #                                            num_workers=nwrks,
    #                                            pin_memory=True,
    #                                            sampler=torch.utils.data.SubsetRandomSampler(train_indices),
    #                                            collate_fn=ctc_collate)

    train_loader = torch.utils.data.DataLoader(htr_dataset_train,
                                           batch_size=train_batch_size,
                                           num_workers=nwrks,
                                           pin_memory=True,
                                           sampler=torch.utils.data.SubsetRandomSampler(train_indices))

    dataset_size = len(htr_dataset_val)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    # val_loader = torch.utils.data.DataLoader(htr_dataset_val,
    #                                          batch_size=val_batch_size,
    #                                          num_workers=nwrks,
    #                                          pin_memory=True,
    #                                          sampler=torch.utils.data.SubsetRandomSampler(indices),
    #                                          collate_fn=ctc_collate)
    
    val_loader = torch.utils.data.DataLoader(htr_dataset_val,
                                             batch_size=val_batch_size,
                                             num_workers=nwrks,
                                             pin_memory=True,
                                             sampler=torch.utils.data.SubsetRandomSampler(indices))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15, min_lr=1e-7)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel with {total_params} parameters to be trained.\n', file=sys.stdout)

    best_val_loss = sys.float_info.max
    last_best_val_epoch = -1
    epochs_without_improving = 0
    predictions_progess = 0

    log_hyperparameters(logs, args, model, optimizer, scheduler, criterion)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        # Unfreeze the encoder after a certain number of epochs
        if frozen and unfreeze > 0 and epoch == unfreeze:
            print(f"Unfreezing encoder weights at epoch {epoch}")
            for param in model.encoder.parameters():
                param.requires_grad = True

        # Train step
        train_loss, train_iou, train_f1, ignored_batches = train_step(model, train_loader, device, criterion, optimizer, verbosity, epoch)
        # continue
        # Validation step
        val_loss, val_iou, val_f1 = val_step(model, val_loader, device, criterion, experiment_dir, epoch)

        # Learning rate adjustment
        scheduler.step(val_loss)
        print(f"LR = {optimizer.param_groups[0]['lr']}")

        # Logging and saving model
        num_img_processed = len(train_loader) - len(ignored_batches)
        logs.write(f"epoch {epoch} train av. loss = {train_loss:.5f} val av. loss = {val_loss:.5f}  train av. IOU = {train_iou:.5f} val av. IOU {val_iou:.5f}  train av. F1 {train_f1:.5f} val av. F1 {val_f1:.5f}\n")
        logs.flush()

        if val_loss < best_val_loss:
            print(f"\033[93m\ttrain av. loss = {train_loss:.5f} val av. loss = {val_loss:.5f}")
            print(f"\033[93m\ttrain av. IOU = {train_iou:.5f} val av. IOU {val_iou:.5f}")
            print(f"\033[93m\ttrain av. F1 = {train_f1:.5f} val av. F1 {val_f1:.5f}")
            best_val_loss = val_loss
            epochs_without_improving = 0

            torch.save({'model': model}, os.path.join(experiment_dir, f'model_{epoch}.pth'))

            if last_best_val_epoch >= 0:
                os.remove(os.path.join(experiment_dir, f"model_{last_best_val_epoch}.pth"))
            last_best_val_epoch = epoch
        else:
            print(f"\ttrain av. loss = {train_loss:.5f} val av. loss = {val_loss:.5f}")
            print(f"\ttrain av. IOU = {train_iou:.5f} val av. IOU {val_iou:.5f}")
            print(f"\ttrain av. F1 = {train_f1:.5f} val av. F1 {val_f1:.5f}")
            epochs_without_improving += 1

        if early_stop and epochs_without_improving > early_stop:
            print(f"Stopping early at epoch {epoch}")
            break
    
    logs.write(f"\nBEST VAL LOSS: epoch {last_best_val_epoch} val av. loss = {best_val_loss:.5f}")
    logs.flush()
    logs.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a model training process of using the given dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_augm', action='store_true', help='enable data augmentation', default=False)
    parser.add_argument('--fixed_width', type=int, help='fixed image width', default=64)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--early_stop', type=int, help='number of epochs without improving', default=-1)
    parser.add_argument('--batch_size', type=int, help='image batch_size', default=24)
    parser.add_argument('--val_batch_size', type=int, help='image batch_size', default=24)
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')     
    parser.add_argument("--verbosity", action="store_true",  help="output verbosity",default=False)
    parser.add_argument('--encoder_name', type=str, help='Encoder to use', default="resnet34")
    parser.add_argument('--encoder_weights', type=str, help='', default="imagenet")
    parser.add_argument('--arch_name', type=str, help='', default="unet")
    parser.add_argument('--frozen', action='store_true', help='freeze encoder weights', default=False)
    parser.add_argument('--lr', type=float, help='image batch_size', default=3e-4)
    parser.add_argument('--unfreeze', type=int, help='image batch_size', default=100)
    parser.add_argument('--base_model', type=str, help='Pre-trained model to finetune', default="")
    

    parser.add_argument('dataset_train', type=str, help='train dataset location')
    parser.add_argument('dataset_val', type=str, help='validation dataset location')
    # parser.add_argument('model_name', type=str, help='Save model with this file name')
    
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.gpu:
        if args.gpu[0] > torch.cuda.device_count() or args.gpu[0] < 0:
            sys.exit(colored("\tERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()),"red"))
        torch.cuda.set_device(args.gpu[0])
          
    # if os.path.isfile(args.model_name):
    #     print("Model already exists")
    #     state = torch.load(args.model_name, map_location=device)
    #     model = state['model']
    if os.path.isfile(args.base_model):
        print(f"Using pretrained model at {args.base_model}")
        state = torch.load(args.base_model, map_location=device)
        model = state['model']
    else:
        print("Initializing model...")
        #model = HTRModel_orig(num_classes= args.fixed_width, line_height=args.fixed_width )
        
        
        # model = HTRModel_0(line_height=args.fixed_width)
        # model = HTRModel_ResNet(line_height=args.fixed_width)
        # model = UNet(n_class=1)

        # model = buildModel(arch_name=args.arch_name, encoder_name=args.encoder_name, encoder_weights=args.encoder_weights)

        # model = UNet(n_class=2, in_channels=1)
        # model = UNet(n_class=1)
        # model = UNet2(num_classes=1, in_channels=1, depth=6)
        model = R2AttU_Net(img_ch=1, output_ch=1)


        # model = smp.Unet(
        #     encoder_name=args.encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights=args.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=1,                      # model output channels (number of classes in your dataset)
        # )
        
        
        #model = HTRModel2D(line_height=args.fixed_width)
        #model = ConvNet(num_classes=args.fixed_width)
        #model = UNet(input_ch=1, n_classes=args.fixed_width)
        model.to(device)
    
       
    if args.verbosity:
        #from pytorch_model_summary import summary
        #print(summary(model,  torch.zeros((args.batch_size, 1, args.fixed_width, args.fixed_width)).cuda(), show_input=True))
        #from torchsummary import summary
        #summary(model.cuda(), (1, args.fixed_width, args.fixed_width))
        #from torchinfo import summary
        #summary(model, (1, args.fixed_width, args.fixed_width), device="cpu")
        
        
        print(model)
        pass
       

    print("\nSelected GPU %i\n"%(torch.cuda.current_device()))
#    img_transforms = procImg.get_tranform(args.fixed_width, args.data_augm)

    train_xy_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, shear=10),  # Add shearing
        T.ElasticTransform(alpha=50.0, sigma=5.0),  # Add elastic deformation
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # Add random erasing
    ])

    train_x_transforms = T.Compose([
        T.RandomInvert(p=0.5),
        T.GaussianBlur(3),
        T.ColorJitter(),
    ])

    """
    Cambiar les transformacions per al set de validació    
    """
    
    
    val_x_transforms = T.Compose([])

    val_xy_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        # T.RandomResizedCrop((768, 768), scale=(0.4, 1.0), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        ])


    # htr_dataset_train = Dataset(args.dataset_train, args.fixed_width, xy_transform=train_xy_transforms, x_transform=train_x_transforms)    
    # htr_dataset_val = Dataset(args.dataset_val, args.fixed_width, xy_transform=val_xy_transforms, x_transform=val_x_transforms)

    htr_dataset_train = Dataset(args.dataset_train, target_size=(512, 512), xy_transform=T.Compose([]), x_transform=T.Compose([]))    
    htr_dataset_val = Dataset(args.dataset_val, target_size=(512, 512), xy_transform=T.Compose([]), x_transform=T.Compose([]))

    print(htr_dataset_train[0][0].shape, htr_dataset_train[0][1].shape, htr_dataset_train[0][2])

    

    start_time=time.time()
    dataset_name = args.dataset_train.split("/")[0]
    
    # if args.base_model:
    #     experiment_dir = create_experiment_folder('experiments', args.arch_name, args.encoder_name, args.encoder_weights, dataset_name, 
    #                                             args.frozen, args.lr, args.unfreeze)
    experiment_dir = create_experiment_folder('experiments', args.arch_name, args.encoder_name, args.encoder_weights, dataset_name, 
                                            args.frozen, args.lr, args.unfreeze, args.base_model)

    
    if args.frozen:
        print("Freezing encoder weights...")
        for param in model.encoder.parameters():
            param.requires_grad = False
            # print(f"{param.requires_grad}")
    else:
        print("Encoder weights not frozen.")

    logs_path = os.path.join(experiment_dir, 'training_logs.txt')
    logs = open(logs_path, 'w')

    log_data_aug(logs, train_xy_transforms, train_x_transforms, val_xy_transforms)
    
    train(model, htr_dataset_train, htr_dataset_val, logs, device, epochs=args.epochs,          
            train_batch_size=args.batch_size, val_batch_size=args.val_batch_size, experiment_dir=experiment_dir, 
            early_stop=args.early_stop, lh=args.fixed_width, verbosity=args.verbosity, lr=args.lr, unfreeze=args.unfreeze, frozen=args.frozen,
        )

    
    # os.system(f"./SRC/test.py --batch_size 1 {experiment_dir} data_cBat_17/test/")

    # test_func(experiment_dir, device, dataset_test, batch_size, )
    

    # print(colored("Saving last model ","green"))
    # torch.save({'model': model, 
    #             'line_height': args.fixed_width}, 
    #             args.model_name.rsplit('.',1)[0]+"_final_train.pth")

    
    logs.close()
    total_time=time.time() - start_time
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    print("Total training time %02i:%02i:%02i"%(h,m,s))
    
    sys.exit(os.EX_OK)