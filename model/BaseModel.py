import torch.nn as nn
from torchvision.models import swin_t, resnet34
import torch
from utils.transform_utils import gen_basis, get_warp_flow


class BaseModel(nn.Module):
    def __init__(self, backbone_1, backbone_2, registration) -> None:
        super().__init__()
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.registration = registration
        self.basis = gen_basis(512, 512).unsqueeze(0).reshape(1, 8, -1)

    def forward(self, batch=None):
        bs, _, h_patch, w_patch = batch["img1_patch"].size()
        img1_patch_fea = self.backbone_1(batch["img1_patch"])
        img2_patch_fea = self.backbone_2(batch["img2_patch"])
        img1_full_fea = self.backbone_1(batch["img1_full"])
        img2_full_fea = self.backbone_2(batch["img2_full"])
        # ========================forward ====================================

        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        weight_f = self.registration(forward_fea)
        H_flow_f = (
            (self.basis.to(forward_fea.device) * weight_f)
            .sum(1)
            .reshape(bs, 2, h_patch, w_patch)
        )
        # 用来转换Img2___get_warp_flow

        # ========================backward===================================
        backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        weight_b = self.registration(backward_fea)
        H_flow_b = (
            (self.basis.to(backward_fea.device) * weight_b)
            .sum(1)
            .reshape(bs, 2, h_patch, w_patch)
        )
        # 用来转换Img1___get_warp_flow

        # warp_img1_patch, warp_img1_patch_fea = list(
        #         map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_patch, img1_patch_fea]))
        # warp_img2_patch, warp_img2_patch_fea = list(
        #     map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_patch, img2_patch_fea]))

        # img1_patch_warp_fea, img2_patch_warp_fea = list(
        #     map(self.fea_extra, [warp_img1_patch, warp_img2_patch]))

        return {
            "H_flow_f": H_flow_f,
            "H_flow_b": H_flow_b,
            "weight_f": weight_f,
            "weight_b": weight_b,
        }


class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的Swin Transformer Tiny模型
        self.backbone = swin_t(weights="IMAGENET1K_V1")

    def forward(self, x):
        # 提取Swin Transformer的中间特征
        features = self.backbone.features(x)
        return features


# class ResBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 使用预训练的Swin Transformer Tiny模型
#         self.backbone = resnet34(weights='IMAGENET1K_V1')

#     def forward(self, x):
#         # 提取Swin Transformer的中间特征

#         ...
#         return features
