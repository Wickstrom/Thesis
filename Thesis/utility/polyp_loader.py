import numpy as np
from torchsample.utils import th_affine2d
from torchsample.transforms import (RandomRotate,
                                    RandomShear,
                                    RandomFlip,
                                    RandomZoom,
                                    AddChannel,
                                    SpecialCrop
                                    )
#######################################################################################
# This fucntion loads a Torch tensor of size (C,H,W) and                              #
# transforms the training data. User must input size to crop image,                   #
# range of rotation, shearing and zoom. The function is not particularly efficient,   #
# but this way ensure that the input and ground truth are transformed in the same way.#
# Note that i use torchsample(https://github.com/ncullen93/torchsample)               #
# for augmentation.                                                                   #
#######################################################################################

def transform(x,y,crop_range,rot_range,shear_range,zoom_range,t):
  
    if t == False:                                               # If t is set to false the input is only cropped
        crop_type = np.random.randint(0,5,1)[0]                  # Randomly crop image from either center or a corner.
    
        x_new = SpecialCrop((crop_range),crop_type=crop_type)(x)
        y_new = SpecialCrop((crop_range),crop_type=crop_type)(y)
    else:
        rot = RandomRotate(rot_range,lazy=True)(x)
        shear = RandomShear(shear_range,lazy=True)(x)
        zoom = RandomZoom(zoom_range,lazy=True)(x)
        flip = RandomFlip(v=True,p = np.random.randint(0,2,1)[0])# Images and label is flipped with 0.5 prob.
        
        crop_type = np.random.randint(0,5,1)[0]
    
        x_new = SpecialCrop((crop_range),crop_type=crop_type)(x)
        y_new = SpecialCrop((crop_range),crop_type=crop_type)(y)
                        
        x_new = th_affine2d(x_new,rot)
        y_new = th_affine2d(y_new,rot)
    
        x_new = th_affine2d(x_new,shear)
        y_new = th_affine2d(y_new,shear)
    
        x_new = th_affine2d(x_new,zoom)
        y_new = th_affine2d(y_new,zoom)
    
        x_new = flip(x_new)
        y_new = flip(y_new)    
        
    return AddChannel()(x_new), AddChannel()(y_new)              # Add channel for concatenating batch.
