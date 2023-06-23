#Pandas
try:
    import pandas as pd 
    print("pandas version-",pd.__version__)
except:
    !pip install pandas -q
    import pandas as pd 
    print("pandas version-",pd.__version__)
#numpy
try:
    import numpy as np
    print("numpy version-",np.__version__)
except:
    !pip install numpy -q
    import numpy as np
    print("numpy version-",np.__version__)
#os,collections,time,path,garbage,tqdm
import os
from collections import defaultdict
import time
import gc
import tqdm
from pathlib import Path
import csv
from copy import *
import math
import numbers
import random
""""#progress bar
try:
    from fastprogress import progessbar
except:
    !pip install fastprogess
    from fastprogress import progessbar"""
#opencv   
try:
    import cv2
    print("opencv version-",cv2.__version__)
except:
    !pip install opencv-python -q
    import cv2
    print("opencv version",cv2.__version__)
#scikit-image
try:
    import skimage
    print("scikit image version",skimage.__version__) 
except:
    !pip install scikit-image -q
    import skimage
    print("scikit image version",skimage.__version__)
#PIL Image
try:
    from PIL import Image,ExifTags
except:
    !pip install PIL
    from PIL import Image,ExifTags
#pycolmap    
try:
    import pycolmap
    print("pycolmap version",pycolmap.__version__)
except:
    !pip install pycolmap -q
    import pycolmap
    print("pycolmap version",pycolmap.__version__)
#mediapy
try:
    import mediapy as media
    print("mediapy version", media.__version__)
except: 
    !pip install mediapy -q
    import mediapy as media
    print("mediapy version", media.__version__)
#glob
from glob import glob

#h5py for hdf5 format
try:
    import h5py
except: 
    !pip install h5py -q
    import h5py
#pytorch & nn functions
try:
    import torch 
    import torch.nn.functional as F
    device=torch.device('cuda')
    print("torch version", torch.__version__)
except: 
    !pip install torch -q
    import torch
    import torch.nn.functionaal as F
    device=torch.device('cuda')
    print("torch version", torch.__version__)
#TIMM
try: 
    import timm
    from timm.data import resolve_data_config
    from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
    from timm.data.random_erasing import RandomErasing
    from timm.data.transforms_factory import create_transform
except:
    !pip install timm
    import timm
    from timm.data import resolve_data_config
    from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
    from timm.data.random_erasing import RandomErasing
    from timm.data.transforms_factory import create_transform
#Kornia
try:
    import kornia 
    print("kornia version", kornia.__version__)
except: 
    !pip install kornia -q
    import kornia
    print("kornia version", kornia.__version__)

    
"""The purpose of the glob_inf function is to extract global information features from a set of images using the provided model. Here's a breakdown of the code:

1.The model is set to evaluation mode and moved to the specified device.
2.The config dictionary is resolved using the resolve_data_config function, providing an empty dictionary and the model as arguments.
3.The transform is created using the create_transform function with the config as arguments.
4.An empty list glob_info_next is initialized to store the extracted global information features.
5.The function iterates over each image file name in the file_names list using tqdm for progress tracking.
6.The base name of the image file is extracted using os.path.basename and split into the key and extension using os.path.splitext.
7.The image is opened using Image.open and converted to RGB format.
8.The image is transformed using the defined transform.
9.The transformed image is unsqueezed to add a batch dimension and moved to the specified device.
10.The model is used to extract features from the transformed image using model.forward_features.
11.The extracted features are averaged over the width and height dimensions using .mean(dim=(-1, 2)).
12.The resulting features are reshaped to have a size of (1, -1).
13.The features are normalized along dimension 1 using F.normalize and the L2 norm.
14.The normalized features are detached from the computational graph and moved to the CPU.
15.The normalized features are appended to the glob_info_next list.
16.The global_info_ tensor is created by concatenating the features in glob_info_next along dimension 0.
17.The global_info_ tensor containing the global information features is returned.
18.The function essentially performs feature extraction on a set of images using a given model and returns the concatenated global information features.





"""
def glob_inf(file_names, model,
                    device =  torch.device('cpu')):
    model = model.eval()
    model= model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    glob_info_next=[]
    for i, img_name in tqdm(enumerate(file_names),total= len(file_names)):
        key = os.path.splitext(os.path.basename(img_name))[0]
        image_ = Image.open(img_name).convert('RGB')
        new_img = transform(image_).unsqueeze(0).to(device)
        with torch.no_grad():
            info = model.forward_features(new_img.to(device)).mean(dim=(-1,2))#
            #print (desc.shape)
            info = info.view(1, -1)
            info_normal = F.normalize(ifno, dim=1, p=2)
        global_info_next.append(info_normal.detach().cpu())
    global_info_ = torch.cat(global_info_next, dim=0)
    return global_info_

"""pairs(file_names):

Parameters:
file_names: A list of file names representing images.
Returns:
pairs: A list of pairs representing unique combinations of indices from the file_names list.
Description:
This function generates pairs of indices from the file_names list. Each pair consists of two indices (i, j) where i and j are different and range from 0 to len(file_names) - 1. The function iterates over the range of indices and appends each unique pair to the pairs list. The resulting pairs list is returned.
pairs_shortlist(file_names, sim_thresh, min_pairs, exhaustiveless, device=torch.device('cpu')):

Parameters:
file_names: A list of file names representing images.
sim_thresh: A similarity threshold value (float) used for pair selection.
min_pairs: The minimum number of pairs to be generated.
exhaustiveless: An integer value specifying the maximum number of images for which exhaustive pair generation will be performed. If the number of images is less than or equal to exhaustiveless, the function falls back to the pairs function.
device: An optional parameter specifying the device on which the model should be loaded. By default, it is set to torch.device('cpu').
Returns:
match_list: A list of pairs representing shortlisted combinations of indices from the file_names list.
Description:
This function generates a shortlist of pairs from the file_names list based on similarity criteria. If the number of images is less than or equal to exhaustiveless, the function directly calls the pairs function to generate pairs. Otherwise, it performs the following steps:
It loads a pre-trained EfficientNet-B7 model using timm.create_model.
The model is set to evaluation mode.
Global information features are extracted from the images using the global_inf function.
Pairwise distances between the extracted features are computed using torch.cdist.
A mask is created based on the similarity threshold.
For each image, matches are obtained based on the mask and minimum pairs requirement.
The resulting matches are filtered and appended to the match_list.
The match_list is sorted and returned as the final shortlisted pairs.
"""


def pairs(file_names):
    pairs = []
    for i in range(len(file_names)):
        for j in range(i+1, len(file_names)):
            pairs.append((i,j))
    return pairs


def pairs_shortlist(file_names,
                              sim_thresh = 0.6, # should be strict
                              min_pairs = 20,
                              exhaustiveless = 20,
                              device=torch.device('cpu')):
    num = len(file_names)

    if num <= exhaustiveless:
        return pairs(file_names)

    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b7/1/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    infos = global_inf(file_names, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    mask = dm <= sim_thresh    
    total = 0
    match_list = []
    ar = np.arange(num)
    already_there = []
    for st in range(num-1):
        mask = mask[st]
        match = ar[mask]
        if len(match) < min_pairs:
            match = np.argsort(dm[st])[:min_pairs]  
        for i in match:
            if st == i:
                continue
            if dm[st, i] < 1000:
                match_list.append(tuple(sorted((st, i.item()))))
                total+=1
    match_list = sorted(list(set(match_list)))
    return match_list