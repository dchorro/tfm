import os, sys
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageDraw
import torchvision.transforms.functional as Fv
import torchvision.transforms as transforms
import numpy as np
from xml.dom import minidom
from collections import OrderedDict
from termcolor import colored
import glob

class Dataset(Dataset):        
    def __init__(self, root_dir, target_size=(512, 512), xy_transform=None, x_transform=None):
        """
        Constructor de la clase Dataset.
        :param root_dir: Directorio raíz donde se encuentran las imágenes y archivos XML.
        :param target_size: Tamaño objetivo para redimensionar las imágenes (por defecto 512x512).
        :param xy_transform: Transformaciones a aplicar tanto a las imágenes como a las anotaciones.
        :param x_transform: Transformaciones a aplicar solo a las imágenes.
        """
        self.root_dir = root_dir
        if not os.path.isdir(root_dir):
            print(colored("\t ERROR: Path \""+root_dir+ "\" does not exist", "red"))
            sys.exit(-1)
            
        files = glob.glob(self.root_dir + "/*.xml")
        self.items = [fname.rsplit('.', 1)[0] for fname in files]
        self.items = list(OrderedDict.fromkeys(self.items))  # Eliminar duplicados manteniendo el orden
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)  # Asegurarse que siempre sea una tupla
        self.sizes = np.empty(len(self.items), dtype=object)

        # Cargar los tamaños originales de las imágenes
        for i, item_name in enumerate(self.items):
            image = Image.open(item_name + ".jpg")
            self.sizes[i] = (image.size[1], image.size[0])  # Guardar el tamaño (alto, ancho)
            image.close()

        self.xy_transform = xy_transform
        self.x_transform = x_transform

    def get_file_name(self, pos):
        """
        Obtiene el nombre del archivo sin extensión en la posición 'pos'.
        """
        return self.items[pos]

    def get_file_size(self, pos):
        """
        Obtiene el tamaño de la imagen original en la posición 'pos'.
        """
        return self.sizes[pos]
    
    def get_dest_img(self, item_name):
        """
        Genera una imagen con las anotaciones dibujadas a partir del archivo XML.
        :param item_name: Nombre base del archivo (sin extensión).
        :return: Imagen con las anotaciones redimensionada al tamaño objetivo.
        """
        # Cargar el archivo XML correspondiente
        if not os.path.isfile(item_name + ".xml"):
            print(colored("\nERROR: file %s not found" % (item_name + ".xml"), "red"))
            sys.exit(-1)
            
        gt_file = minidom.parse(item_name + ".xml")
        base_lines = gt_file.getElementsByTagName('Baseline')  # Extraer las líneas base
        points = []
        
        # Extraer y convertir los puntos de las líneas base
        for line in base_lines:
            l = line.attributes['points'].value.split()
            line = []
            for point in l:
                x = int(point.split(",")[0])
                y = int(point.split(",")[1])
                line.append(x)
                line.append(y)
            points.append(line)

        # Obtener las dimensiones originales de la imagen desde el XML
        pag = gt_file.getElementsByTagName('Page')
        imageWidth = int(pag[0].attributes['imageWidth'].value)
        imageHeight = int(pag[0].attributes['imageHeight'].value)
        
        size_orig = (imageWidth, imageHeight)  # Tamaño original de la imagen
        size_dest = self.target_size  # Tamaño destino fijo (512x512, por ejemplo)
        
        # Crear una nueva imagen en blanco con el tamaño objetivo directamente
        im = Image.new('L', size=size_dest, color=0)
        
        # Escalar los puntos de anotación para que se ajusten al tamaño destino
        self.scale_points(points, size_orig, size_dest)

        # Dibujar las líneas en la imagen redimensionada
        draw = ImageDraw.Draw(im)
        for line in points:
            draw.line(line, fill=255, width=0)

        return im

    def scale_points(self, points, size_orig, size_dest):
        """
        Escalar los puntos de anotación para ajustarlos al nuevo tamaño de imagen.
        :param points: Lista de puntos originales.
        :param size_orig: Tamaño original de la imagen.
        :param size_dest: Tamaño destino.
        """
        for line in points:
            for p in range(len(line)):
                if p % 2 == 0:  # Coordenada x
                    line[p] = line[p] * size_dest[0] // size_orig[0]
                else:  # Coordenada y
                    line[p] = line[p] * size_dest[1] // size_orig[1]

    def __len__(self):
        """
        Devuelve el número de elementos en el dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.items[idx]
        
        # Abre la imagen y asegúrate de que existe
        if not os.path.isfile(item_name + ".jpg"):
            print(colored("\nERROR file %s not found" % (item_name + ".jpg"), "red"))
            sys.exit(-1)

        # Abre la imagen y conviértela a escala de grises
        image = Image.open(item_name + ".jpg").convert('L')
        
        # Redimensiona la imagen al tamaño deseado
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Obtén la imagen de ground truth (gt_image)
        gt_image = self.get_dest_img(item_name)
        
        # Convierte ambas imágenes (input y gt) a tensores
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        gt_image = toTensor(gt_image)

        return image, gt_image, idx


def ctc_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    idxs = [item[2] for item in batch]


    x = zero_pad(x)
    y = zero_pad(y)
    x = torch.stack(x) # x ---> Tensor:NxCxHxW'
    y = torch.stack(y)
   
    return x, y, idxs


def zero_pad(x):
    # x is a list of N tensors, each tensor has shape CxHxW_i, where W_i is the seq length of the i-th sample
    max_w = max(xi.shape[2] for xi in x)
    max_c = max(xi.shape[0] for xi in x)
    max_h = max(xi.shape[1] for xi in x)
    
    shape = (max_c, max_h, max_w)
    # shape is now CxHxW', where C is max_c, H is max_h, and W' is max_w

    out = []
    for xi in x:
        o = torch.zeros(shape)
        # this ensures that the padding is applied correctly to tensors of different sizes
        o[:xi.shape[0], :xi.shape[1], :xi.shape[2]] = xi
        out.append(o)

    # out is now a list of N tensors, each tensor has shape CxHxW'
    return out







"""

# Standard packages
import os,sys
import torch
from torch.utils.data import Dataset
from PIL import Image,ImageOps,ImageDraw

#from torchvision.utils import save_image
import torchvision.transforms.functional as Fv
import torchvision.transforms as transforms

import numpy as np
#import xmltodict
from xml.dom import minidom
from collections import OrderedDict
#from procImg import get_tranform
from termcolor import colored
import glob
import torchvision.transforms as T
# import torchvision.transforms.v2 as v2


class Dataset(Dataset):        
    def __init__(self, root_dir, width, xy_transform=None, x_transform=None):
        self.root_dir = root_dir
        if not os.path.isdir(root_dir):
            print(colored("\t ERROR: Path \""+root_dir+ "\" does not exists", "red"))
            sys.exit(-1)
            
        files = glob.glob(self.root_dir+"/*.xml")

        #files = os.listdir(self.root_dir)
        self.items = [fname.rsplit('.',1)[0] for fname in files]
        self.items =  list(OrderedDict.fromkeys(self.items))
        self.sizes = np.empty(len(self.items), dtype=object)
        for i, item_name in enumerate(self.items):
             image = Image.open(item_name+".jpg")
             self.sizes[i] = (image.size[1], image.size[0])
             image.close()             

        self.xy_transform = xy_transform
        self.x_transform = x_transform
        self.width = width

    def get_file_name(self,pos):
        return self.items[pos]

    def get_file_size(self, pos):
        return self.sizes[pos]
    
    def calc_new_height(self, old_size):
        old_width, old_height = old_size
        aspect_ratio = old_height / old_width
        return round(self.width * aspect_ratio)
    
    def get_dest_img(self, item_name):
        #get baselines
        if not os.path.isfile(item_name+".xml"):
            print(colored("\nERROR file %s not found"%(item_name+".xml"),"reed"))
            sys.exit(-1)
            
        gt_file = minidom.parse(item_name+".xml")
        
        base_lines = gt_file.getElementsByTagName('Baseline')
        points=[]
        for line in base_lines:
            l=line.attributes['points'].value.split()
            line = []
            for point in l:
                x=int(point.split()[0].split(",")[0])
                y=int(point.split()[0].split(",")[1])
                line.append(x)
                line.append(y)
            points.append(line)

        pag = gt_file.getElementsByTagName('Page')
        imageWidth = int(pag[0].attributes['imageWidth'].value)
        imageHeight = int(pag[0].attributes['imageHeight'].value)
        print(f"Image width: ", imageWidth)
        print(f"Image height: ", imageHeight)
        
        size_orig = (imageWidth, imageHeight)
        size_dest = (self.width, self.calc_new_height(size_orig))
        
        # size_dest = (self.width, self.calc_new_height(size_orig)//2)
#        size_dest=(self.width, self.width)
        im = Image.new('L', size=size_orig, color=0)
        # im = Image.new('L', size=size_dest, color=0)
        
        # self.scale_points(points, size_orig, size_dest)

        draw = ImageDraw.Draw(im)
        for line in points:
            draw.line(line, fill = 255,  width = 0)
      
        #im[im > 0.0] = 1.0

        # im = Fv.rotate(im,90,expand=True)#,Image.NEAREST)
        
        return im


    def scale_points(self, points,size_orig,size_dest):
        for line in points:
           # print(len(line))
            for p in range(len(line)):
                if p % 2 != 0:
                    line[p] = line[p] * size_dest[1] // size_orig[1]
                else:
                    line[p] = line[p] * size_dest[0] // size_orig[0]


    # A custom Dataset class must implement __len__ function
    def __len__(self):
        return len(self.items)

    # A custom Dataset class must implement __getitem__ function
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item_name = self.items[idx]
        #image = Image.open(os.path.join(self.root_dir, item_name + ".jpg"))
        if not os.path.isfile(item_name+".jpg"):
            print(colored("\nERROR file %s not found"%(item_name+".jpg"),"reed"))
            sys.exit(-1)
            
        image = Image.open(item_name+".jpg")
        image = image.convert('L')
        
        # new_size = (self.calc_new_height(image.size),self.width)
        # new_size = (self.width,self.width)
        # image = Fv.resize(image, new_size ,antialias=Image.Resampling.LANCZOS)
        
        # Rotate image horizontally
        # image = Fv.rotate(image,90,expand=True)

        # Black background & White letters
        # image = ImageOps.invert(image)
                
        gt_image = self.get_dest_img(item_name)
        
        # XOR operation -> No pot ser que uno siga True i l'altre False
        if bool(self.xy_transform) ^ bool(self.x_transform):
            raise Exception("No es posible que una de las dos transformaciones sea None")
        elif self.xy_transform:
            state = torch.get_rng_state()
            gt_image = self.xy_transform(gt_image)
            torch.set_rng_state(state)
            new_t = transforms.Compose([self.xy_transform, self.x_transform])
            image = new_t(image)
        else:
            toTensor = transforms.Compose([transforms.ToTensor()])
            image = toTensor(image)
            gt_image = toTensor(gt_image)

      #  black_pixels=(gt_image == 1).sum()
      #  per_cent_black_pixels=black_pixels/(gt_image.shape[1] * gt_image.shape[2])
      #  print("\nper_cent "+str(per_cent_black_pixels.item()))
        return image, gt_image, idx
        



# This is required by the DataLoader
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/docs/stable/data.html
def ctc_collate(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    idxs = [item[2] for item in batch]


    x = zero_pad(x)
    y = zero_pad(y)
    x = torch.stack(x) # x ---> Tensor:NxCxHxW'
    y = torch.stack(y)
   
    return x, y, idxs


# Reference:
#   N: mini bach size
#   C: number of channels
#   H: height of feature maps
# W_i: width of the ith feature map
#
# For each mini-batch, this function add zeros to all the samples sequences 
# whose lengths were lesser than the sample sequence of maximum length in 
# that mini-batch. So, all the sample sequences will have the same length.
# This is required by the above "ctc_collate" function


def zero_pad(x):
    # x is a list of N tensors, each tensor has shape CxHxW_i, where W_i is the seq length of the i-th sample
    max_w = max(xi.shape[2] for xi in x)
    max_c = max(xi.shape[0] for xi in x)
    max_h = max(xi.shape[1] for xi in x)
    
    shape = (max_c, max_h, max_w)
    # shape is now CxHxW', where C is max_c, H is max_h, and W' is max_w

    out = []
    for xi in x:
        o = torch.zeros(shape)
        # this ensures that the padding is applied correctly to tensors of different sizes
        o[:xi.shape[0], :xi.shape[1], :xi.shape[2]] = xi
        out.append(o)

    # out is now a list of N tensors, each tensor has shape CxHxW'
    return out

"""