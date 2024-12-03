from segment_anything import sam_model_registry, SamPredictor
from dataset import SAMAugmentedDataset,BaseDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from model.net import getDemoModel
from utils.config_utils import Params
from tqdm import tqdm
from loss.losses import LossL1,LossL2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import cv2
from model.DHN import DHN
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ =="__main__":
    max_epoches=200
    param=Params("/CV/gaobiaoli/project/RegistrationNet/config/param.json")
    device='cuda'
    # dataset = BaseDataset(image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train",ratio=0.1,patch_size=128)
    dataset = BaseDataset(image_dir="/CV/gaobiaoli/dataset/coco/train2017",ratio=0.01,patch_size=128)
    dataloader=DataLoader(dataset=dataset,batch_size=64,num_workers=8)
    model=DHN()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    l2l=nn.MSELoss(reduction='mean')
    for epoch in range(max_epoches):
        total_loss=0
        total_num=0
        with tqdm(total=len(dataloader), ncols=100,desc=f'Epoch-{epoch+1}') as t:
            for id,batch in enumerate(dataloader):
                target=batch['disturbed_corners'].to(device)
                input=torch.cat([batch['img1_patch'],batch['img2_patch']],dim=1).to(device)
                pred=model(input)
                loss=l2l(target.flatten(1),pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=( loss.item() * len(batch['img1_patch']) )
                total_num+=len(batch['img1_patch'])
                t.set_postfix({"loss":loss.item(),"LR":scheduler.get_last_lr()[0]})
                t.update()  
        scheduler.step()
        print(f"Loss:{total_loss/total_num}")
