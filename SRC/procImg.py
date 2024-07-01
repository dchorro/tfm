# Standard packages
import torchvision.transforms as transforms
import torchvision.transforms.functional as Fv

from PIL import ImageOps,Image

class FixedHeightResize(object):
    """
    from https://github.com/pytorch/vision/issues/908
    """
    def __init__(self, height):
        self.height = height

    def __call__(self, img):       
        size = (self.height, self._calc_new_width(img))
        #size = (self.height,256)
        #size = (self.height,self.height)        
        return Fv.resize(img, size ,antialias=Image.Resampling.LANCZOS)
    
    def _calc_new_width(self, img):
        old_width, old_height = img.size#img.shape[1],img.shape[2]
#        print(old_width,old_height)
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)
    
class RotateImg(object):
    def __init__(self,angle=90):
        self.angle=angle

    def __call__(self,img):
        return Fv.rotate(img,self.angle,expand=True)#,Image.NEAREST)

# Transformations applied for image processing
# https://pytorch.org/vision/main/transforms.html
def get_tranform(fixedHeight=64, dataAug=False, ref=False):

    # List of transformations for Data Augmentation
    daTr = [
             # Random AdjustSharpness
             #transforms.RandomAdjustSharpness(1, p=0.5),
             # Random Rotation
             #transforms.RandomRotation(1,fill=255),
             # Random Affine Transformations (this includes Rotation)
             transforms.RandomAffine(degrees=1,
                                     translate=(0.005,0.05),
                                     shear=1,
                                     fill=255),
             # Random Gaussian Blur
             # transforms.GaussianBlur(3, sigma=(0.001, 1.0))
           ] if dataAug else []

    # List of mandatory transformations
    nTr = [        
#        RotateImg(),
        # Invert pixel values
        transforms.Lambda(lambda x: ImageOps.invert(x)),
        # Convert a (PIL) image to tensor                 
#        FixedHeightResize(fixedHeight)
        #transforms.ToTensor()
        ] 
        

    return transforms.Compose( daTr + nTr )
