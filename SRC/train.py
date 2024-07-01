#!/usr/bin/env python3

# Standard packages
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

def buildModel(arch_name, encoder_name, encoder_weights, in_channels=1, n_classes=1):
    if arch_name == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
    elif arch_name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
    elif arch_name == "deeplabv3p":
        # print("Architecture not implemented yet.")
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


def create_experiment_folder(base_dir, arch_name, encoder_name, encoder_weights):
    # Create a folder name based on current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{current_time}_{arch_name}_{encoder_name.replace('-','_')}_{encoder_weights}"

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


def train(model, htr_dataset_train, htr_dataset_val, device, train_batch_size, val_batch_size, experiment_dir, epochs=20, early_stop=10, lh=64, verbosity=False):
    # To control the reproducibility of the experiments
    torch.manual_seed(17)

    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores
    train_loader = torch.utils.data.DataLoader(htr_dataset_train,
                                            batch_size = train_batch_size,
                                            num_workers=nwrks,
                                            pin_memory=True,
                                            shuffle = True, 
                                            # shuffle = False, 
                                            collate_fn = ctc_collate)
    
    val_loader = torch.utils.data.DataLoader(htr_dataset_val,
                                             batch_size = val_batch_size,
                                             num_workers=nwrks,
                                             pin_memory=True,
                                             shuffle = True,
                                            #  shuffle = False, 
                                             collate_fn = ctc_collate)


    # criterion = get_loss("dice")
    # criterion = get_loss("weightbce")
    # criterion = nn.BCEWithLogitsLoss()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=3e-6, cooldown=5)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)


    '''
    # Print model state_dict
    print("Model state_dict:",file=sys.stderr)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file=sys.stderr)
    # Print optimizer state_dict
    print("Optimizer state_dict:", file=sys.stderr)
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name], file=sys.stderr)
    '''
    # Print the total number of parameters of the model to train
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel with {total_params} parameters to be trained.\n', file=sys.stdout)

    # Epoch loop
    best_val_loss=sys.float_info.max
    last_best_val_epoch=-1
    epochs_without_improving=0
    predictions_progess = 0

    logs_path = os.path.join(experiment_dir, 'training_logs.txt')
    logs = open(logs_path, 'w')

    for epoch in range(epochs):
        total_train_loss = 0
        ignored_batches=list()
        batch_num=0

        model.train()
        
        try:
            terminal_cols = os.get_terminal_size()[0]
        except IOError:
            terminal_cols = 80
            
        format='{l_bar}{bar:'+str(terminal_cols-48)+'}{r_bar}'

       
        # Mini-Batch train loop    
        print("Epoch %i"%(epoch))
        for (x, y, bIdxs) in tqdm(train_loader, bar_format=format, colour='green', desc='  Train'):
            batch_num = batch_num + 1
            torch.cuda.empty_cache()

            x, y = x.to(device), y.to(device)
            x, _ = pad_to_32(x)
            y, _ = pad_to_32(y)
            
            # foreground_pixels = y.sum(dim=[1,2,3])
            # # print(foreground_pixels.shape)
            # print(foreground_pixels.item())
            # print(y.shape[2]*y.shape[3])
            # print("-------------------")
            # print(foreground_pixels.item()/(y.shape[2]*y.shape[3]))
            # print("-------------------")

            # weights = foreground_pixels / y.shape[3]
            # weights = weights.permute(0, 2, 1)
            
            outputs = model(x)

            if verbosity and batch_num == 1 and epoch < 5:
                # print("-------------------------")
                # print(x.shape)
                # print(y.shape)
                # print(outputs.shape)
                # print("-------------------------")


                # print(f"Background %: {foreground_pixels / y.shape[3]}, {foreground_pixels} / {y.shape[3]}")
                # out_ = outputs.unsqueeze(0)     #C,W,B,H
                # out_ = outputs.permute(2, 0, 3, 1)   # B,C,H,W
                # thres =  0.8 #threshold_otsu(out_[0][0])
                #               out_ = out_ > thres
                out_ = (outputs > 0.5).float()
                outputs_ =  make_grid(out_, nrow=2,normalize=True,scale_each=True)
                save_image(outputs_, f"outputs_{predictions_progess}.png",nrows=1, normalize=True)
            
                x_ = make_grid(x, nrow=2, normalize=True, scale_each=True)
                save_image(x_,f"out_x_{predictions_progess}.png",nrow=1)
            
                y_ = nnf.interpolate(y, size=(y.size(2),int(y.size(3))), mode='bicubic', align_corners=False)  
                y_ =  make_grid(y_,nrow=2,normalize=True, scale_each=True)
                save_image(y_, f"out_y_{predictions_progess}.png", nrow=1)
                predictions_progess += 1
            
            
            loss = criterion(outputs, y)
            # loss = criterion(outputs, y, weights)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
        
            total_train_loss += loss.item()

        if len(ignored_batches) > 0 :
            percent_batches="{:2.2f}".format(100*len(ignored_batches)/batch_num)
            print(colored("  "+ str(len(ignored_batches))+" Ignored batches  ("+percent_batches+"% of total)","green"))
            if verbosity:
                print(colored(" "+files,"green"))


        # --------------- VAL ---------------
        model.eval()
        
        with torch.no_grad():
            val_loss = 0
            first = True
            for (x,y,bIdxs) in tqdm(val_loader, bar_format=format, colour='magenta', desc='  Valid'):
                torch.cuda.empty_cache()
                x, y = x.to(device), y.to(device)
                x, _ = pad_to_32(x)
                y, _ = pad_to_32(y)

                outputs = model(x)

                if first and epoch >= 30 and epoch % 5 == 0:
                    out_ = (outputs[:2, :, : ,:] > 0.5).float()
                    outputs_ =  make_grid(out_, nrow=2,normalize=True,scale_each=True)
                    save_image(outputs_, os.path.join(experiment_dir, f"outputs_{epoch}.png"), nrows=1, normalize=True)
                
                    x_ = make_grid(x[:2, :, : ,:], nrow=2, normalize=True, scale_each=True)
                    save_image(x_, os.path.join(experiment_dir, f"out_x_{epoch}.png"), nrow=1)
                
                    y_ = nnf.interpolate(y[:2, :, : ,:], size=(y.size(2),int(y.size(3))), mode='bicubic', align_corners=False)  
                    y_ =  make_grid(y_,nrow=2,normalize=True,scale_each=True)
                    save_image(y_, os.path.join(experiment_dir, f"out_y_{epoch}.png"), nrow=1)
                    first = False


                loss = criterion(outputs, y)
                # loss = criterion(outputs, y, weights)
                val_loss += loss.item()
    
            scheduler.step(val_loss)
            print(f"LR = {optimizer.param_groups[0]['lr']}")
        num_img_processed = len(train_loader) - (len(ignored_batches) * val_batch_size) #len(bIdxs))
       # print ("\ttrain av. loss = %.5f val av. loss = %.5f"%(total_train_loss/num_img_processed,val_loss/len(val_loader)))
      
        if num_img_processed == 0:
            print(colored("\ttrain av. loss = INF val av. loss = INF","red"))
            continue
        
        if (val_loss  < best_val_loss):
            print ("\033[93m\ttrain av. loss = %.5f val av. loss = %.5f\x1b[0m"%(total_train_loss/num_img_processed,val_loss/len(val_loader)))
            logs.write("epoch %i train av. loss = %.5f val av. loss = %.5f\n"%(epoch, total_train_loss/num_img_processed,val_loss/len(val_loader)))
            logs.flush()
            epochs_without_improving=0
            best_val_loss = val_loss
            torch.save({'model': model}, os.path.join(experiment_dir, f'model_{str(epoch)}.pth'))
            
            
            if os.path.exists(os.path.join(experiment_dir, f'model_{str(last_best_val_epoch)}.pth')):
                os.remove(os.path.join(experiment_dir, f"model_{str(last_best_val_epoch)}.pth"))
            last_best_val_epoch=epoch
        else:
            print ("\ttrain av. loss = %.5f val av. loss = %.5f"%(total_train_loss/num_img_processed, val_loss/len(val_loader))) 
            epochs_without_improving = epochs_without_improving + 1
                
        if early_stop > 0 and epochs_without_improving >= early_stop:
            print(colored("\nEarly stoped after %i epoch without improving"%(early_stop),"green"))
            return 


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
    
    parser.add_argument('dataset_train', type=str, help='train dataset location')
    parser.add_argument('dataset_val', type=str, help='validation dataset location')
    parser.add_argument('model_name', type=str, help='Save model with this file name')
    
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.gpu:
        if args.gpu[0] > torch.cuda.device_count() or args.gpu[0] < 0:
            sys.exit(colored("\tERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()),"red"))
        torch.cuda.set_device(args.gpu[0])
          
    if os.path.isfile(args.model_name):
        print("Model already exists")
        state = torch.load(args.model_name, map_location=device)
        model = state['model']
    else:
        print("Initializing model...")
        #model = HTRModel_orig(num_classes= args.fixed_width, line_height=args.fixed_width )
        
        
        # model = HTRModel_0(line_height=args.fixed_width)
        # model = HTRModel_ResNet(line_height=args.fixed_width)
        # model = UNet(n_class=1)


        model = buildModel(args.arch_name, args.encoder_name, args.encoder_weights)

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
       

    print("\nSelected GPU %i\n"%(torch.cuda.current_device()))
#    img_transforms = procImg.get_tranform(args.fixed_width, args.data_augm)

    # Llevar els horizontal i vertical flip
    train_xy_transforms = T.Compose([
        T.ToTensor(),
        # T.Resize((256,256), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomResizedCrop((512,512), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        # T.RandomHorizontalFlip(p=0.2),
        # T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(10),
        ])

    # Llevar invert
    train_x_transforms = T.Compose([
        T.RandomInvert(p=0.5),
        T.GaussianBlur(3),  
        T.ColorJitter(),
    ])

    htr_dataset_train = Dataset(args.dataset_train, args.fixed_width, xy_transform=train_xy_transforms, x_transform=train_x_transforms)


    htr_dataset_val = Dataset(args.dataset_val, args.fixed_width)

    start_time=time.time()
    experiment_dir = create_experiment_folder('experiments', args.arch_name, args.encoder_name, args.encoder_weights)

    # try:
    train(model, htr_dataset_train, htr_dataset_val, device, epochs=args.epochs,          
            train_batch_size=args.batch_size, val_batch_size=args.val_batch_size, experiment_dir=experiment_dir, 
            early_stop=args.early_stop, lh=args.fixed_width, verbosity=args.verbosity)
    # except KeyboardInterrupt as err:     
    #     pass
#    finally:
#        print(colored("Saving last model ","green"))
#        torch.save({'model': model, 
#                    'line_height': args.fixed_width}, 
#                   args.model_name.rsplit('.',1)[0]+"_final_train.pth")

    logs.close()
    total_time=time.time() - start_time
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    print("Total training time %02i:%02i:%02i"%(h,m,s))
    
    sys.exit(os.EX_OK)
