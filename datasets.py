import os
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import glob
from utils.pallete import *
import re

import torch
class NYUDepth(Dataset):
    def __init__(self, path_img, path_target, transforms):
        self.path_img = path_img
        with open(path_target, 'rb') as f:
            self.targets = pickle.load(f)
        self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        target = self.targets[index]

        
        img = Image.open(os.path.join(self.path_img, target['name']))
        anno = scipy.io.loadmat(os.path.join(self.path_img, target['name']).replace('/train','/train_anno_mat').replace('.png','.mat'))['anno'].astype('uint8')
        anno = Image.fromarray(anno)
        if self.transforms:
            num_targets = len(target['x_A'])
            landmarks = np.zeros((num_targets*2,2))

            landmarks[:num_targets,0] = target['x_A']
            landmarks[:num_targets,1] = target['y_A']
            landmarks[num_targets:,0] = target['x_B']
            landmarks[num_targets:,1] = target['y_B']
            
            img,anno,landmarks = self.transforms(img,anno,landmarks)
            
            lm = np.hstack([landmarks[:num_targets,:],landmarks[num_targets:,:],target['ordinal_relation'][:,np.newaxis]])

            ind1 = np.where(lm[:,:3]<0)[0]
            ind2 = np.where(lm[:,[0,2]]>=img.shape[1])[0]
            ind3 = np.where(lm[:,[1,3]]>=img.shape[2])[0]
            #ind = np.concatenate([ind1,ind2,ind3])
            #ind = np.unique(ind)
            #print(ind1.shape,ind2.shape,ind3.shape,ind.shape)
            #np.logical_or(np.logical_or(lm[:,0]<0,lm[:,1]<0),np.logical_or(np.logical_or(lm[:,2]<0,lm[:,3]<0),
            #np.logical_or(np.logical_or(lm[:,0]>=img.shape[0],lm[:,2]>=img.shape[0]),np.logical_or(lm[:,1]>=img.shape[1],lm[:,3]>=img.shape[1])))))
            #landmarks = np.delete(lm,ind,0)
            landmarks = lm
            landmarks[ind1,4] = 2
            landmarks[ind2,4] = 2
            landmarks[ind3,4] = 2

            target = {}
            target['x_A'] = landmarks[:,0]
            target['y_A'] = landmarks[:,1]
            target['x_B'] = landmarks[:,2]
            target['y_B'] = landmarks[:,3]
            target['ordinal_relation'] = landmarks[:,4]
        
            
        return img, anno, target

class NYUSeg(Dataset):
    def __init__(self, path_img, path_target, transforms):
        self.path_img = path_img
        self.imgs = sorted(glob.glob(path_img+'/*.jpg'))
        self.lbls = sorted(glob.glob(path_target+'/*.png'))
        self.sp = path_target
        #with open(path_target, 'rb') as f:
        #    self.targets = pickle.load(f)
        #self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        #target = self.targets[index]

        
        img = Image.open(self.imgs[index])
        # anno = cv2.imread(self.imgs[index].replace('/train/','/train_labels_13/').replace('/test/','/test_labels_13/').replace('nyu_rgb_','new_nyu_class13_'),cv2.IMREAD_UNCHANGED)
        #anno = Image.open(self.lbls[index])
        
        anno = cv2.imread(self.lbls[index], cv2.IMREAD_UNCHANGED)-1

        anno = Image.fromarray(anno)

        img_name = re.findall(r'\d+', self.imgs[index])[0]
        anno_name = re.findall(r'\d+', self.lbls[index])[0]
        # print(self.imgs[index])
        # print(img_name)
        if int(img_name) != int(anno_name):
            print("Anamoly")
        #if self.tolabel is True:
        #    anno = get_mask_pallete(np.array(anno), dataset=self.dataset)
        
        # img_np = np.array(img)
        # if np.isnan(img_np).any():
        #     print("Nan Value Image")
        # anno = np.array(anno)-1
        #print(self.imgs[index],np.unique(anno))
        # anno = Image.fromarray(anno)
        if self.transforms:
            
            img,anno,landmarks = self.transforms(img,anno,landmarks = None)

            target = {}
            target['x_A'] = []
            target['y_A'] = []
            target['x_B'] = []
            target['y_B'] = []
            target['ordinal_relation'] = []
        
        return img, anno, target



     
