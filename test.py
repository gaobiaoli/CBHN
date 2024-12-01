import cv2
import numpy as np
import torch
from PIL import Image
from utils.transform_utils import generateTrainImagePair,get_warp_flow

img_path="/CV/gaobiaoli/project/RegistrationNet/demo/demo.jpg"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img,img_warp, flow, disturbed_corners,ori, H=generateTrainImagePair(image,image)

get_warp_flow(torch.tensor(img_warp).unsqueeze(0).permute(0,3,1,2),torch.tensor(flow).unsqueeze(0).permute(0,3,1,2),start=[100,100])