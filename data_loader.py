from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset

from PIL import Image
#==========================dataset load==========================


class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform_img=None,transform_lbl=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform_img = transform_img
		self.transform_lbl = transform_lbl

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = io.imread(self.image_name_list[idx]).astype(np.uint8)
		if len(image.shape) == 2:
			image = np.stack([image] * 3, axis=-1)
		if len(image.shape) == 3 and image.shape[0] == 1:
			image = np.repeat(image, 3, axis=0)
		name = self.image_name_list[idx]

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx]).astype(np.uint8)

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		sample = {'image':Image.fromarray(image), 'label':Image.fromarray(label, mode="L"), 'name':name}

		if self.transform_img:
			image = sample['image']
			image = self.transform_img(image)
		if self.transform_lbl:
			label = sample['label']
			label = self.transform_lbl(label)
		name = sample['name']
		sample = {'image':image, 'label':label, 'name':name}

		return sample