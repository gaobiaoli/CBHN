from segment_anything import sam_model_registry, SamPredictor
from dataset import SAMAugmentedDataset,BaseDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from model.net import getDemoModel
from utils.config_utils import Params
from tqdm import tqdm
from loss.losses import LossL1,LossL2,TripletLoss
from utils.train_utils import tensor_gpu
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
    return LossL2()(flow_4cor,tgt_cor4)

def cal_loss(pred,batch):
    mse_loss=cal_mace(batch['disturbed_corners'],pred['H_flow_1'])
    positive = pred['img1_patch_warp_fea']
    anchor = pred['img2_patch_fea']
    negative = pred['img1_patch_fea']

    tri_forward_loss = TripletLoss()(anchor=anchor, positive=positive, negative=negative).mean()
    total_loss=mse_loss+tri_forward_loss
    return {"total_loss":total_loss,"mace":torch.sqrt(mse_loss/8)}
if __name__ =="__main__":
    max_epoches=200
    param=Params("/CV/gaobiaoli/project/RegistrationNet/config/param.json")
    device='cuda'
    # sam = sam_model_registry['vit_b'](checkpoint="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth").to(device)
    dataset = SAMAugmentedDataset(image_dir="/CV/gaobiaoli/dataset/CIS-Dataset/train", ratio=0.1,patch_size=param.crop_size[0],reshape=(1024,1024))
    dataloader=DataLoader(dataset=dataset,batch_size=2,num_workers=6,pin_memory=True,shuffle=True,drop_last=True)
    model=getDemoModel(param=param)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for epoch in range(max_epoches):
        total_loss=0
        total_num=0
        with tqdm(total=len(dataloader), ncols=100,desc=f'Epoch-{epoch+1}') as t:
            for id,batch in enumerate(dataloader):
                batch=tensor_gpu(batch)
                pred=model(batch)
                # loss=l2l(pred['H_flow_f'],flow_img2)+l2l(pred['H_flow_b'],flow_img1)
                # loss=0
                # loss=cal_mace(batch['disturbed_corners'],pred['H_flow_b'])
                loss=cal_loss(pred,batch)
                optimizer.zero_grad()
                loss['total_loss'].backward()
                optimizer.step()
                total_loss+=loss['mace']*len(batch['img1_full'])
                total_num+=len(batch['img1_full'])
                t.set_postfix({"loss":loss['total_loss'].item(),"LR":scheduler.get_last_lr()})
                t.update()  
        scheduler.step()
        print(f"Loss:{total_loss/total_num}")
