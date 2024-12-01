from segment_anything import sam_model_registry, SamPredictor
from dataset import SAMAugmentedDataset
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
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
def cal_mace(tgt_cor4,pred_flow):

    flow_4cor = torch.zeros((tgt_cor4.shape[0], 2, 2, 2)).to(tgt_cor4.device)
    flow_4cor[:,:, 0, 0]  = pred_flow[:,:, 0, 0]
    flow_4cor[:,:, 0, 1] = pred_flow[:,:,  0, -1]
    flow_4cor[:,:, 1, 0] = pred_flow[:,:, -1, 0]
    flow_4cor[:,:, 1, 1] = pred_flow[:,:, -1, -1]
    return F.mse_loss(flow_4cor,tgt_cor4)
if __name__ =="__main__":
    max_epoches=10
    param=Params("/CV/gaobiaoli/project/RegistrationNet/config/param.json")
    device='cuda'
    sam = sam_model_registry['vit_b'](checkpoint="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth").to(device)
    dataset = SAMAugmentedDataset(image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train", sam=sam,ratio=0.1)
    dataloader=DataLoader(dataset=dataset,batch_size=8,num_workers=8,pin_memory=True,shuffle=True)
    model=getDemoModel(param=param)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=param.gamma)
    l2l=LossL2()
    for epoch in range(max_epoches):
        total_loss=0
        total_num=0
        with tqdm(total=len(dataloader), ncols=100,desc=f'Epoch-{epoch+1}') as t:
            for id,batch in enumerate(dataloader):
                # batch['img1'] = batch['img1'].to(device)
                # batch['img2'] = batch['img2'].to(device)
                target=batch['disturbed_corners'].to(device)
                flow_img1=batch['flow_img1'].to(device)
                flow_img2=batch['flow_img1'].to(device)
                pred=model(batch['img1'].to(device),batch['img2'].to(device))
                # loss=l2l(pred['H_flow_f'],flow_img2)+l2l(pred['H_flow_b'],flow_img1)
                # loss=0
                loss=cal_mace(target,pred['H_flow_f'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()
                total_num+=len(batch['img1'])
                t.set_postfix({"loss":loss.item(),"LR":scheduler.get_last_lr()})
                t.update()  
        scheduler.step()
        print(f"Loss:{total_loss/total_num}")
