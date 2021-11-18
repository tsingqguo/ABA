import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from pysot.utils.bbox import center2corner, Center , Corner ,corner2center 
import math
# ImageNet code should change this value
PI=3.1415926

def center_ro(cx,cy,px,py,a):
    a=-2*PI/360*a
    x= (px - cx)*math.cos(a) - (py - cy)*math.sin(a) + cx
    y= (px - cx)*math.sin(a) + (py - cy)*math.cos(a) + cy

    return x , y

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _,IMAGE_SIZE,bbox):
  return ImageOps.autocontrast(pil_img) ,bbox


def equalize(pil_img, _,IMAGE_SIZE,bbox):
  return ImageOps.equalize(pil_img) ,bbox


def posterize(pil_img, level,IMAGE_SIZE,bbox):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level) ,bbox


def rotate(pil_img, level,IMAGE_SIZE,bbox):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  px,py,_ ,_ = corner2center(bbox)# 原始矩形角点
  a,b,c,d = bbox
  #print(bbox)
  cenx,ceny = center_ro(pil_img.size[0]/2,pil_img.size[1]/2, px ,  py,degrees)
  po2x,po2y = center_ro(pil_img.size[0]/2,pil_img.size[1]/2, a , d,degrees)
  po3x,po3y = center_ro(pil_img.size[0]/2,pil_img.size[1]/2, c , d,degrees)
  hh=max(abs(po2y-ceny)*2,abs(po3y-ceny)*2)
  ww=max(abs(po3x-cenx)*2,abs(po2x-cenx)*2)
  bbox = center2corner(Center(cenx,ceny,ww,hh))

  #print(bbox)
  
  return pil_img.rotate(degrees, resample=Image.BILINEAR) , bbox


def solarize(pil_img, level,IMAGE_SIZE,bbox):
  level = int_parameter(sample_level(level), 256)




  return ImageOps.solarize(pil_img, 256 - level) , bbox

def paste(im,level,IMAGE_SIZE,bbox):
  #print(type(im))
  #img = Image.fromarray(im)      np.array(im).transpose(2,0,1)
  avg = np.array(im).sum()/IMAGE_SIZE/IMAGE_SIZE/3
  mk = np.ones((10,10,3))*avg
  mk = Image.fromarray(np.uint8(mk))
  x = int(np.random.uniform(low=45,high=IMAGE_SIZE-55))
  y = int(np.random.uniform(low=45,high=IMAGE_SIZE-55))
  im.paste(mk,(x,y))
  return im, bbox
   
def shear_x(pil_img, level,IMAGE_SIZE,bbox):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  #print(bbox)
  x,y,w,h = corner2center(bbox)
  x-=y*level
  w = w*(1+abs(0.5*level))
  bbox = center2corner(Center(x,y,w,h))
  #print(bbox)
  
  

  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR) , bbox


def shear_y(pil_img, level,IMAGE_SIZE,bbox):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  #print(bbox)
  x,y,w,h = corner2center(bbox)
  y-=x*level
  h = h * (1+abs(2*level))
  bbox = center2corner(Center(x,y,w,h))
  #print(bbox)
  
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR) , bbox



def translate_x(pil_img, level,IMAGE_SIZE,bbox):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR) , bbox



def translate_y(pil_img, level,IMAGE_SIZE,bbox):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR) , bbox



# operation that overlaps with ImageNet-C's test set
def color(pil_img, level,IMAGE_SIZE,bbox):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level) , bbox



# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level,IMAGE_SIZE,bbox):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level) , bbox



# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level,IMAGE_SIZE,bbox):
    level = float_parameter(sample_level(level), 1.9) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level) , bbox



# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level,IMAGE_SIZE,bbox):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level) , bbox


#def scale()

augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]
augmentations_augmix = [
     rotate,  shear_x, shear_y,  autocontrast, equalize, posterize,  solarize, 
     color, contrast, brightness, sharpness
]#scale，
augmentations_feature = [
    autocontrast, equalize, posterize,  solarize, 
     color, contrast, brightness, sharpness
]
augmentations_duo = [
  paste, shear_x, shear_y, rotate, contrast , translate_x, translate_y , brightness
  
  
]
