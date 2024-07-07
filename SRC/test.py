#!/usr/bin/env python3

# Standard packages
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import sys, os, argparse
from tqdm import tqdm
import torch
import numpy as np
#import cupy as np
from torchvision.utils import save_image, make_grid
from multiprocessing import cpu_count
from termcolor import colored
import time
import torch.nn.functional as nnf
import torchvision.transforms.functional as Fv
import torchvision.transforms as T
from PIL import Image, ImageDraw
from xml.dom import minidom
import math
import time

# Local packages
import procImg
#from model import HTRModel
from dataset_p2xml import Dataset, ctc_collate
from UFSet import DisjointSet

Image.MAX_IMAGE_PIXELS = None #to avoid decompression bomb DOS attack warning

# FUNCTION THAT EXECUTES JAVA -JAR TranskribusBaseLineEvaluationScheme for each folder and saves the result inside it
def create_eval_file(subfolder):
    os.popen(f"ls {subfolder}/*.xml > hyp.lst")
    os.popen(f"java -jar TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar gt.lst hyp.lst > {subfolder}/evaluation.txt").read()


def extract_and_write_lines(input_file, model_name):
    # Define the output file names
    
    # Open the input file and read its content
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Find the lines that need to be extracted
    p_line = None
    r_line = None
    f1_line = None
    
    for line in lines:
        if "Avg (over pages) P value" in line:
            p_line = line.strip().split(": ")[-1]
        elif "Avg (over pages) R value" in line:
            r_line = line.strip().split(": ")[-1]
        elif "Resulting F_1 value" in line:
            f1_line = line.strip().split(": ")[-1]
    
    # Ensure we found all necessary lines
    if p_line and r_line and f1_line:
        # Write each line to its respective file
        # print("------------------------")
        # print("ALL THREE LINES WERE FOUND!")
        # print("------------------------")
        # print()
        with open("table_P.txt", 'a') as p_file:
            p_file.write(f"{model_name:50}: {p_line}\n")
        
        with open("table_R.txt", 'a') as r_file:
            r_file.write(f"{model_name:50}: {r_line}\n")
        
        with open("table_F1.txt", 'a') as f1_file:
            f1_file.write(f"{model_name:50} {f1_line}\n")
    else:
        print("Error: Could not find all the required lines in the input file.")



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

def test(model, dataset_test, device, out_dir, batch_size=1, verbosity=False):
    # To control the reproducibility of the experiments
    nwrks = int(cpu_count() * 0.70) # use ~2/3 of avilable cores   
    
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                             batch_size = batch_size,
                                             num_workers=nwrks,
                                             pin_memory=True,
                                             shuffle = False, 
                                             collate_fn = ctc_collate)
    

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

    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    try:
        terminal_cols = os.get_terminal_size()[0]
    except IOError:
        terminal_cols = 80
            
    format='{l_bar}{bar:'+str(terminal_cols-48)+'}{r_bar}'
        
    model.eval()
    test_loss = 0
    with torch.no_grad():        
        for (x,y,bIdxs) in tqdm(test_loader, bar_format=format,dynamic_ncols=True, colour='magenta', desc='  Test'):
            #torch.cuda.empty_cache()


            x = x.to(device)
            y = y.to(device)

            x, _ = pad_to_32(x)
            y, _ = pad_to_32(y)
            # print(x.shape)
                
            outputs = model(x)
            # print(y)
            test_loss += criterion(outputs, y)

            # outputs = (outputs > 0.5).float()
            # class_1 = y.sum()
            # class_1_weight = (class_1/np.prod(y.shape)).item() 
            # print("\n", class_1_weight)
            # print(y.shape, np.prod(y.shape))

            # tp, fp, fn, tn = smp.metrics.get_stats(outputs, y.type(torch.LongTensor).to(device), mode='binary', threshold=0.5)
            # print(f"tp:{tp}, fp:{fp}, fn:{fn}, tn:{tn}")
            # f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            # f1_score_weighted = smp.metrics.f1_score(tp, fp, fn, tn, reduction="weighted", class_weights=[class_1_weight, 1-class_1_weight])
            # print(f"\nF1 Score: {f1_score}")
            # print(f"\nF1 Score: {f1_score_weighted}")


            out_ = outputs
            for i in range(out_.shape[0]):
                out_file_name = os.path.basename(dataset_test.get_file_name(bIdxs[i]))
                out = (out_[i] > 0.5).float()
                # out = out_[i]

                # outputs_ =  make_grid(out, nrow=2, normalize=True, scale_each=True)
                # save_image(outputs_, out_dir+"/"+out_file_name+"_hyp.png", nrows=1, normalize=True)

                # y_ = y[i].squeeze()
                # y_ =  make_grid(y_, nrow=2, normalize=True, scale_each=True)
                # save_image(y_, out_dir+"/"+out_file_name+"_ref.png", nrows=1, normalize=True)
               
               
                # HARD-CODED TO TEST FOLDER
                # print(out_file_name)
                get_lines(out, "data/test"+"/"+out_file_name, device, write_folder=out_dir)
        
        
        print(test_loss)

                
def get_lines(image, file_name, device, write_folder):
#    image=image.to(device)
    # print("----------------------------------------------")
    # print(image.shape)
    # print(image.shape)
    # print("----------------------------------------------")
    
    image = image.squeeze()
    image = image.cpu().numpy()
    
    # image = image[0,:,:]
    rows, cols = image.shape
    
    n_pixels = rows * cols
    ufset = DisjointSet(n_pixels)

    # t_ufset_0 = time.time()
    for f in range(rows):
        for c in range(cols):
            pixel = image[f][c].item()
           
            if pixel > 0.2:            
                if c+1 < cols and image[f,c+1].item() > 0.2:
                    ufset.union(f*cols + c, f*cols + c+1)                

                if f+1 < rows and c+1 < cols and image[f+1,c+1].item() > 0.2:
                    ufset.union(f*cols + c, (f+1)*cols + c+1)
                    
                    
                if f+1 < rows and image[f+1,c].item() > 0.35:
                    ufset.union(f*cols + c, (f+1)*cols + c)
                    
    linies = {}
#    t_ufset = time.time() - t_ufset_0
#    print("\ntemps ufset create "+str(t_ufset))
 
    for i,_ in enumerate(ufset.parent):
        if i == ufset.parent[i] and ufset.rank[i] > 0:
            linies[i]=[]
    for f in range(rows):
        for c in range(cols):
            rep=ufset.find(f*cols + c)
            #if rank[rep] > 0:
            if rep in linies:
                punts = linies[rep]
                punts.append((c,f))
                linies[rep] = punts

    # print("----------------------------------------------")
    # print(len(linies))
    # # print(linies)
    # print("----------------------------------------------")
   
    
    xml_file = minidom.parse(file_name+".xml")
    pag=xml_file.getElementsByTagName('Page').item(0)
    text_region = xml_file.createElement("TextRegion")
    text_region.setAttribute('id','r0')

    coords_region = xml_file.createElement("Coords")
    coords_region.setAttribute('points',"0,0,0,0")
    text_region.appendChild(coords_region)

    for line_num, l in enumerate(linies):

        text_line = xml_file.createElement("TextLine")
        text_line.setAttribute('id', 'r0_l%02d'%line_num) 

        coords_line = xml_file.createElement("Coords")
        coords_line.setAttribute("points","")
        text_line.appendChild(coords_line)

        base_line = xml_file.createElement("Baseline")
        
        val=linies[l]
        val.sort(key=lambda x: x[0])
        map_rep={}
        val_no_rep=[]
        
        for punt in val:
            if punt[0] not in map_rep:
                map_rep[punt[0]]=punt[1]               
            else:
                y=map_rep[punt[0]]
                if image[y][punt[0]] < image[punt[1]][punt[0]]:
                    map_rep[punt[0]]=punt[1]


        for punt in val:
            if punt[0] in map_rep and map_rep[punt[0]] == punt[1]: 
                val_no_rep.append((punt[0],punt[1]))

        val_no_rep=normaliza_traza(val_no_rep,15)
        
        punts_str = ' '.join(str(x)+","+str(y) for x,y in val_no_rep)        
        base_line.setAttribute("points",punts_str)
        text_line.appendChild(base_line)
    
        text_region.appendChild(text_line)
     #   dr.line(val_no_rep,fill="white")
    
    pag.appendChild(text_region)
    xml_str = xml_file.toprettyxml(indent ="\t")
 
    with open(str(write_folder+"/"+(file_name.split("/")[-1]))+".xml", "w") as f: 
        f.write(xml_str)





# Create XML structure from the existing file
    xml_file = minidom.parse(file_name + ".xml")
    pag = xml_file.getElementsByTagName('Page').item(0)
    text_region = pag.getElementsByTagName("TextRegion").item(0)
    
    # Clear existing TextLine and Coords elements
    while text_region.firstChild:
        text_region.removeChild(text_region.firstChild)
    
    coords_region = xml_file.createElement("Coords")
    coords_region.setAttribute('points', "0,0,0,0")
    text_region.appendChild(coords_region)

    for line_num, l in enumerate(linies):
        text_line = xml_file.createElement("TextLine")
        text_line.setAttribute('id', 'r0_l%02d' % line_num)
        
        coords_line = xml_file.createElement("Coords")
        coords_line.setAttribute("points", "")
        text_line.appendChild(coords_line)
        
        base_line = xml_file.createElement("Baseline")
        val = linies[l]
        val.sort(key=lambda x: x[0])
        map_rep = {}
        val_no_rep = []
        
        for punt in val:
            if punt[0] not in map_rep:
                map_rep[punt[0]] = punt[1]
            else:
                y = map_rep[punt[0]]
                if image[y][punt[0]] < image[punt[1]][punt[0]]:
                    map_rep[punt[0]] = punt[1]

        for punt in val:
            if punt[0] in map_rep and map_rep[punt[0]] == punt[1]:
                val_no_rep.append((punt[0], punt[1]))

        val_no_rep = normaliza_traza(val_no_rep, 15)
        punts_str = ' '.join(str(x) + "," + str(y) for x, y in val_no_rep)
        base_line.setAttribute("points", punts_str)
        text_line.appendChild(base_line)
        text_region.appendChild(text_line)

    xml_str = xml_file.toprettyxml(indent="\t")

    with open(str(write_folder + "/" + (file_name.split("/")[-1])) + ".xml", "w") as f:
        f.write(xml_str)




    #im.save(file_name+"_line.jpg")
    

def normaliza_traza(baseline, num_points):
   # print(lines)

#    new_lines=[]
#    for baseline in lines:
    if (len(baseline) <= 0):
        return baseline

    baseline_dist=[]
    baseline_dist.append(0)
    for p in range(1,len(baseline)-1):
        DX=(baseline[p][0] - baseline[p-1][0]);
        DY=(baseline[p][1] - baseline[p-1][1]);
        baseline_dist.append(baseline_dist[p-1]+ math.sqrt((DX*DX+DY*DY)))
        
          
    dist_between_points = baseline_dist[len(baseline_dist)-1] / (num_points - 1)
    
    new_baseline=[]
    new_baseline.append(baseline[0])
    n=1
    for p in range(1,num_points-1):        
        while (n < len(baseline_dist)-1  and (baseline_dist[n] < p*dist_between_points)):     
            n=n+1
        if n >= len(baseline_dist) -1:
            continue
        
        C=(p*dist_between_points-baseline_dist[n-1]) / (baseline_dist[n]-baseline_dist[n-1]); #porcentage del segmento en el que caería el punto
            
        TX=baseline[n-1][0] + (baseline[n][0] - baseline[n-1][0])*C
        TY=baseline[n-1][1] + (baseline[n][1] - baseline[n-1][1])*C
        new_baseline.append((int(TX+0.5), int(TY+0.5)))
    
    new_baseline.append(baseline[len(baseline)-1])
    return new_baseline


def process_subfolders(parent_folder):
    # List all subfolders in the parent folder
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    # print(subfolders)

    for subfolder in subfolders[:1]:
        # Look for .pth files in the current subfolder
        for filename in os.listdir(subfolder):
            if filename.endswith(".pth"):
                # Construct the full file path
                pth_file = os.path.join(subfolder, filename)
                print(pth_file)
                
                # Execute the function foo and get the result
                result = foo()  # Replace with any logic you need

                # Save the result in the same subfolder
                with open(os.path.join(subfolder, "result.txt"), 'w') as f:
                    f.write(result)

                # Break after processing one .pth file per subfolder
                break

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a model training process of using the given dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, help='image batch_size', default=24)
    parser.add_argument('--gpu', type=int, default=0, help='used gpu')    

    parser.add_argument("--verbosity", action="store_true",  help="output verbosity",default=False)

    parser.add_argument('parent_folder', type=str, help='Parent folder')
    parser.add_argument('dataset_test', type=str, help='validation dataset location')
    # parser.add_argument("out_dir", type=str, help="outputs to dir",default=None)
   
    
    
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # process_subfolders("experiments/")
    

    if args.gpu:
        if args.gpu > torch.cuda.device_count() or args.gpu < 0:
            sys.exit(colored("\tERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()),"red"))
        torch.cuda.set_device(args.gpu)    
          
    # if os.path.isfile(args.model_name):        
    #     state = torch.load(args.model_name, map_location=device)
    #     model = state['model']
    #     line_width = state['line_height']
    # else:
    #     print("Model "+args.model_name+ " does not exitsts" )
    #     sys.exit()


    print("\nSelected GPU %i\n"%(torch.cuda.current_device()))

#    img_transforms = procImg.get_tranform( line_height)


    dataset_test = Dataset(args.dataset_test, 400)
#                                 transform=img_transforms)


    start_time=time.time()

    # test(model, dataset_test, device, batch_size=args.batch_size, out_dir=args.out_dir, verbosity=args.verbosity)

    parent_folder = args.parent_folder
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    # print(subfolders)

    # subfolder = "experiments/2024-06-25_18-50_unetpp_mobileone_s4_imagenet/"
    # filename = "model_47.pth"
    # with torch.no_grad():
    #     torch.cuda.empty_cache()
    #     pth_file = os.path.join(subfolder, filename)
    #     print(pth_file)

    #     state = torch.load(pth_file, map_location=device)
    #     model = state['model']
    
    #     # Execute the function foo and get the result
    #     test(model, dataset_test, device, batch_size=args.batch_size, out_dir=subfolder, verbosity=args.verbosity)

    # print(subfolders)
    a = "Model_Name"
    output_files = {
                    "table_P.txt": f"{a:50} P Value\n",
                    "table_R.txt": f"{a:50} R Value\n",
                    "table_F1.txt": f"{a:50} F1 Value\n"
                }

    for file_name, header in output_files.items():
        with open(file_name, 'w') as file:
            file.write(header)

    for _, subfolder in enumerate(subfolders[:]):
        # Look for .pth files in the current subfolder
        for filename in os.listdir(subfolder):
            if filename.endswith(".pth"):
                # Construct the full file path
                pth_file = os.path.join(subfolder, filename)
                evaluation_file = os.path.join(subfolder, "evaluation.txt")

                model_name = "_".join(subfolder.split("_")[2:])

                
                # create_eval_file(subfolder)


                extract_and_write_lines(evaluation_file, model_name)
                
                # print(len(res))
                # print(res[-90:])
                # print()
                # print(res)


                
                # with torch.no_grad():
                    
                #     torch.cuda.empty_cache()
                #     state = torch.load(pth_file, map_location=device)
                #     model = state['model']
                
                #     # Execute the function foo and get the result
                #     test(model, dataset_test, device, batch_size=args.batch_size, out_dir=subfolder, verbosity=args.verbosity)

                    # Save the result in the same subfolder
                    # with open(os.path.join(subfolder, "result.txt"), 'w') as f:
                    #     f.write(result)
                # Break after processing one .pth file per subfolder
                break



    total_time=time.time() - start_time
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    print("Total time %02i:%02i:%02i"%(h,m,s))
    
    sys.exit(os.EX_OK)

