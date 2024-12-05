import cv2
import numpy as np
import torch
from PIL import Image
from utils.transform_utils import generateTrainImagePair,get_warp_flow
from dataset import BaseDataset,SAMAugmentedDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.train_utils import tensor_gpu
from segment_anything import sam_model_registry, SamPredictor
from utils.visual_utils import show_tensor,display_images_in_row,show_mask
device="cuda"
sam = sam_model_registry['vit_b'](checkpoint="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth").to(device)
dataset = SAMAugmentedDataset(sam=sam,image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train",rho=1,ratio=0.01,patch_size=512,reshape=(1024,1024),keep_ratio=False)

item=dataset[1]
i1_p=dataset.denormlize(item['img1_patch'].permute(1,2,0).numpy()).astype(np.uint8)
i2_p=dataset.denormlize(item['img2_patch'].permute(1,2,0).numpy()).astype(np.uint8)
i1_f=dataset.denormlize(item['img1_full'].permute(1,2,0).numpy()).astype(np.uint8)
i2_f=dataset.denormlize(item['img2_full'].permute(1,2,0).numpy()).astype(np.uint8)
# display_images_in_row([i1_p,show_mask(i2_p)])
# display_images_in_row([i1_p,i2_p])
# display_images_in_row([cv2.addWeighted(i1_p,0.5,show_mask(i2_p),0.5,1)])
display_images_in_row([cv2.addWeighted(i1_f,0.5,show_mask(i2_f),0.5,1)]).save("./demo.png")

sam = sam_model_registry['vit_b'](checkpoint="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth").to(device)
dataset = SAMAugmentedDataset(sam=sam,image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train",rho=0,ratio=0.01,patch_size=512,reshape=(1024,1024),keep_ratio=False)
item=dataset[1]
i1_p=dataset.denormlize(item['img1_patch'].permute(1,2,0).numpy()).astype(np.uint8)
i2_p=dataset.denormlize(item['img2_patch'].permute(1,2,0).numpy()).astype(np.uint8)
i1_f=dataset.denormlize(item['img1_full'].permute(1,2,0).numpy()).astype(np.uint8)
i2_f=dataset.denormlize(item['img2_full'].permute(1,2,0).numpy()).astype(np.uint8)
# display_images_in_row([i1_p,show_mask(i2_p)])
# display_images_in_row([i1_p,i2_p])
# display_images_in_row([cv2.addWeighted(i1_p,0.5,show_mask(i2_p),0.5,1)])
# display_images_in_row([cv2.addWeighted(i1_f,0.5,show_mask(i2_f),0.5,1)])
display_images_in_row([i1_f,show_mask(i2_f)]).save("./demo1.png")
display_images_in_row([cv2.addWeighted(i1_f,0.5,show_mask(i2_f),0.5,1)]).save("./demo2.png")