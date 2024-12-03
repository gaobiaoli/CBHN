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
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import cv2
from model.SwinHN import SwinHN
from torchvision.models import SwinTransformer
from utils.train_utils import tensor_gpu

def cal_mace(tgt_cor4,pred_flow):
    
    flow_4cor = torch.zeros((tgt_cor4.shape[0], 2, 2, 2)).to(tgt_cor4.device)
    flow_4cor[:,:, 0, 0]  = pred_flow[:,:, 0, 0]
    flow_4cor[:,:, 0, 1] = pred_flow[:,:,  0, -1]
    flow_4cor[:,:, 1, 0] = pred_flow[:,:, -1, 0]
    flow_4cor[:,:, 1, 1] = pred_flow[:,:, -1, -1]
    return LossL2()(flow_4cor,tgt_cor4)

if __name__ =="__main__":
    max_epoches=100
    param=Params("/CV/gaobiaoli/project/RegistrationNet/config/param.json")
    device='cuda'
    # dataset = SAMAugmentedDataset(image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train", sam=sam,ratio=0.1,patch_size=128)
    dataset = BaseDataset(image_dir="/CV/gaobiaoli/dataset/coco/train2017",ratio=1,patch_size=128)
    param.crop_size=128
    dataloader=DataLoader(dataset=dataset,batch_size=32,num_workers=8)
    model=getDemoModel(param=param)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    l2l=LossL2()
    for epoch in range(max_epoches):
        total_loss=0
        total_num=0
        with tqdm(total=len(dataloader), ncols=100,desc=f'Epoch-{epoch+1}') as t:
            for id,batch in enumerate(dataloader):
                tensor_gpu(batch)
                target=batch['disturbed_corners'].to(device)
                input=torch.cat([batch['img1_patch'],batch['img2_patch']],dim=1).to(device)
                pred=model(batch)
                loss=cal_mace(batch['disturbed_corners'],pred['H_flow_1'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=( loss.item() * len(batch['img1_patch']) )
                total_num+=len(batch['img1_patch'])
                t.set_postfix({"loss":loss.item(),"LR":scheduler.get_last_lr()[0]})
                t.update()  
        scheduler.step()
        print(f"Loss:{total_loss/total_num}")
