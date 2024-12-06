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
sam = sam_model_registry['vit_b'](checkpoint="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth").to(device=device)

dataset = SAMAugmentedDataset(sam=sam,image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train",rho=16,ratio=1,patch_size=512,reshape=(1024,1024),keep_ratio=False)
dataset.generateSAMCache(start=2133)
# dataset[2133]
