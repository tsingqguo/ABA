import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np
#from common_path import train_set_path_ as dataset_dir

def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1) # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor
def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor
def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1,2,0)
    img = tensor.cpu().numpy().clip(0,255).astype(np.uint8)
    return img


class GOT10k_dataset(Dataset):
    def __init__(self, max_num=15):
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('list.txt')  #init_gt
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
    def __getitem__(self, index):
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'*.jpg')))
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        init_tensor = img2tensor(init_frame_arr)
        init_tensor = torch.nn.functional.interpolate(init_tensor, size=(127,127), mode='bilinear')
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num+1] # to avoid being out of GPU memory
        num_search = len(search_region_paths)-1
        search_tensor1 = torch.zeros((num_search,3,255,255),dtype=torch.float32)
        search_tensor0 = torch.zeros((num_search,3,255,255),dtype=torch.float32)
        for i in range(num_search):
            search_arr1 = cv2.imread(search_region_paths[i+1])
            search_arr0 = cv2.imread(search_region_paths[i])
            #print(search_arr.shape)
            search_arr1 = img2tensor(search_arr1)
            search_arr0 = img2tensor(search_arr0)
            search_tensor1[i] = torch.nn.functional.interpolate(search_arr1, size=(255,255), mode='bilinear')
            search_tensor0[i] = torch.nn.functional.interpolate(search_arr0, size=(255,255), mode='bilinear')
           
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        return (init_tensor, search_tensor0,search_tensor1)
    def __len__(self):
        return len(self.folders_list)


class GOT10k_dataset_dimp(Dataset):
    def __init__(self, max_num=15):
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('list.txt')  #init_gt
        self.folders_list = [os.path.join(dataset_dir,folder) for folder in folders]
        self.max_num = max_num
    def __getitem__(self, index):
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder,'*.jpg')))

        gt_file = os.path.join(cur_folder,'groundtruth.txt')
        gt_arr = np.loadtxt(gt_file,dtype=np.float64,delimiter=',')
        #print(gt_arr)
        
        '''merge init gt into one file'''
        bbox = gt_arr[0].copy()
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        init_tensor = img2tensor(init_frame_arr)
        #init_tensor = torch.nn.functional.interpolate(init_tensor, size=(127,127), mode='bilinear')
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num+1] # to avoid being out of GPU memory
        num_search = len(search_region_paths)-1
        search_tensor1 = torch.zeros((num_search,3,init_tensor.size(2),init_tensor.size(3)),dtype=torch.float32)
        search_tensor0 = torch.zeros((num_search,3,init_tensor.size(2),init_tensor.size(3)),dtype=torch.float32)
        #search_tensor0=[]
        #search_tensor1=[]
        for i in range(num_search):
            search_arr1 = cv2.imread(search_region_paths[i+1])
            search_arr0 = cv2.imread(search_region_paths[i])
           
            #print(search_arr.shape)
            search_arr1 = img2tensor(search_arr1)
            search_arr0 = img2tensor(search_arr0)
            #print(search_arr0.size())
            
            search_tensor1[i] = search_arr1
            search_tensor0[i] = search_arr0
            #search_tensor1[i] = torch.nn.functional.interpolate(search_arr1, size=(255,255), mode='bilinear')
            #search_tensor0[i] = torch.nn.functional.interpolate(search_arr0, size=(255,255), mode='bilinear')
           
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        return (init_frame_arr, search_tensor0 , search_tensor1 ,bbox)
    def __len__(self):
        return len(self.folders_list)



