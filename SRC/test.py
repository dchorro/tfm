#!/usr/bin/env python3

# Standard packages
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

    try:
        terminal_cols = os.get_terminal_size()[0]
    except IOError:
        terminal_cols = 80
            
    format='{l_bar}{bar:'+str(terminal_cols-48)+'}{r_bar}'
        
    model.eval()
    with torch.no_grad():        
        for (x,y,bIdxs) in tqdm(test_loader, bar_format=format,dynamic_ncols=True, colour='magenta', desc='  Test'):
            #torch.cuda.empty_cache()


            x = x.to(device)
                
            outputs = model(x)
            out_ = outputs.unsqueeze(0)     #C,W,B,H               
            out_ = out_.permute(2, 0, 3, 1) #B,C,H,W

            for i in range(out_.shape[0]):
                out_file_name = os.path.basename(dataset_test.get_file_name(bIdxs[i]))
                size_= dataset_test.get_file_size(bIdxs[i])
                out = Fv.rotate(out_[i],-90, expand=True)
                out = Fv.resize(out,size_)

                y = Fv.rotate(y,-90, expand=True)
                save_image(y, out_dir+"/"+out_file_name+"_y.jpg", nrows=1, normalize=True)
                y = Fv.resize(y,size_)

                save_image(out, out_dir+"/"+out_file_name+"_hyp.jpg", nrows=1, normalize=True)
                save_image(y, out_dir+"/"+out_file_name+"_ref.jpg", nrows=1, normalize=True)
               # get_lines(out,out_dir+"/"+out_file_name, device)

                
def get_lines(image, file_name, device):
#    image=image.to(device)
    image = image.cpu().numpy()
    image = image[0,:,:]
    
    rows, cols = image.shape
    
    n_pixels=rows * cols
    ufset=DisjointSet(n_pixels)  

#    threshold = threshold_otsu(image)
    #image = image > 50 #threshold   

 #   t_ufset_0 = time.time()
    for f in range(rows):
        for c in range(cols):
            pixel=image[f][c].item()
           
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
                punts=linies[rep]
                punts.append((c,f))
                linies[rep]=punts
   
    
    im = Image.new("RGB", (cols,rows))
#    dr = ImageDraw.Draw(im)
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
 
    with open(str(file_name)+".xml", "w") as f: 
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
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a model training process of using the given dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, help='image batch_size', default=24)
    parser.add_argument('--gpu', type=int, default=0, help='used gpu')    

    parser.add_argument("--verbosity", action="store_true",  help="output verbosity",default=False)

    parser.add_argument('model_name', type=str, help='Model')
    parser.add_argument('dataset_test', type=str, help='validation dataset location')
    parser.add_argument("out_dir", type=str, help="outputs to dir",default=None)
   
    
    args = parser.parse_args()
    print ("\n"+str(sys.argv)+"\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.gpu:
        if args.gpu > torch.cuda.device_count() or args.gpu < 0:
            sys.exit(colored("\tERROR: gpu must be in the rang of [0:%i]"%(torch.cuda.device_count()),"red"))
        torch.cuda.set_device(args.gpu)    
          
    if os.path.isfile(args.model_name):        
        state = torch.load(args.model_name, map_location=device)
        model = state['model']
        line_width = state['line_height']
    else:
        print("Model "+args.model_name+ " does not exitsts" )
        sys.exit()


    print("\nSelected GPU %i\n"%(torch.cuda.current_device()))

#    img_transforms = procImg.get_tranform( line_height)


    dataset_test = Dataset(args.dataset_test,line_width)
#                                 transform=img_transforms)


    start_time=time.time();

    test(model, dataset_test, device, batch_size=args.batch_size, out_dir=args.out_dir,verbosity=args.verbosity)

    total_time=time.time() - start_time;
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)

    print("Total time %02i:%02i:%02i"%(h,m,s))
    
    sys.exit(os.EX_OK)
