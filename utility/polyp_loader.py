import numpy as np
from torchsample.utils import th_affine2d
from torchsample.transforms import (RandomRotate,
                                    RandomShear,
                                    RandomFlip,
                                    RandomZoom,
                                    AddChannel,
                                    SpecialCrop
                                    )
def transform(x,y,crop_range,rot_range,shear_range,zoom_range,t):
  
    
    if t ==False:
        crop_type = np.random.randint(0,5,1)[0]
    
        x_new = SpecialCrop((crop_range),crop_type=crop_type)(x)
        y_new = SpecialCrop((crop_range),crop_type=crop_type)(y)
        
    else:
        rot = RandomRotate(rot_range,lazy=True)(x)
        shear = RandomShear(shear_range,lazy=True)(x)
        zoom = RandomZoom(zoom_range,lazy=True)(x)
        flip = RandomFlip(v=True,p = np.random.randint(0,2,1)[0])
        
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
        
    return AddChannel()(x_new), AddChannel()(y_new)
