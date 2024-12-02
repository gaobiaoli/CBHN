import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from SAM.SamAutomaticMaskGenerator import SamAutomaticMaskGenerator
from utils.transform_utils import generateTrainImagePair


class BaseDataset(Dataset):
    def __init__(
        self,
        image_dir,
        cache_dir="cache",
        device="cuda",
        transform=None,
        ratio=1.0,
        reshape=(640,640),
        keep_ratio=True,
        patch_size=328
    ):
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_files = self.image_files[0 : int(len(self.image_files) * ratio)]
        self.device = device
        self.transform = transform
        self.cache_dir = cache_dir
        self.patch_size=patch_size
        self.reshape=reshape
        self.keep_ratio=keep_ratio
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
    def __len__(self):
        return len(self.image_files)

    def _process_mask(self, mask):
        # 检查mask的维度并做相应处理
        if len(mask.shape) == 2:  # (w, h)
            mask = mask.unsqueeze(0)  # (1, w, h)
        elif len(mask.shape) == 3:
            if mask.shape[2] == 1:  # (w, h, 1)
                mask = mask.squeeze(2)  # (w, h)
                mask = mask.unsqueeze(0)  # (1, w, h)
            elif mask.shape[2] == 3:  # (w, h, 3)
                mask = mask.permute(2, 0, 1)  # (3, w, h)

        return mask
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        
        # 单应性变换
        data_dict = generateTrainImagePair(image, image, patch_size=self.patch_size,reshape=self.reshape,keep_ratio=self.keep_ratio)

        # 将图像和扭曲后的mask转换为tensor
        data_dict['img1_patch'] = torch.from_numpy(self.normlize(data_dict['img1_patch'])).permute(2, 0, 1).float()
        data_dict['img2_patch'] = self._process_mask(torch.from_numpy(self.normlize(data_dict['img2_patch']))).float()
        data_dict['img1_full'] = torch.from_numpy(self.normlize(data_dict['img1_full'])).permute(2, 0, 1).float()
        data_dict['img2_full'] = self._process_mask(torch.from_numpy(self.normlize(data_dict['img2_full']))).float()
        data_dict['disturbed_corners'] = torch.from_numpy(data_dict['disturbed_corners']).permute(2, 0, 1).float()

        # 如果有其他transform，进行处理
        # if self.transform:
        #     image = self.transform(image)
        
        return data_dict
    
    def normlize(self,ndarray):
        return (ndarray-self.mean_I)/self.std_I
    
class SAMAugmentedDataset(Dataset):
    def __init__(
        self,
        image_dir,
        sam,
        cache_dir="cache",
        device="cuda",
        transform=None,
        ratio=1.0,
        patch_size=328
    ):
        # 初始化图像文件夹路径，SAM模型，设备等参数
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_files = self.image_files[0 : int(len(self.image_files) * ratio)]
        self.device = device
        self.transform = transform
        self.cache_dir = cache_dir
        self.patch_size=patch_size
        self.autoMask = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.2,
            stability_score_thresh=0.8,
            stability_score_offset=2,
            crop_nms_thresh=0.5,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        # 创建缓存文件夹
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_files)

    def _process_mask(self, mask):
        # 检查mask的维度并做相应处理
        if len(mask.shape) == 2:  # (w, h)
            mask = mask.unsqueeze(0)  # (1, w, h)
        elif len(mask.shape) == 3:
            if mask.shape[2] == 1:  # (w, h, 1)
                mask = mask.squeeze(2)  # (w, h)
                mask = mask.unsqueeze(0)  # (1, w, h)
            elif mask.shape[2] == 3:  # (w, h, 3)
                mask = mask.permute(2, 0, 1)  # (3, w, h)

        return mask

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.open(img_path))
        # # 获取图像的embedding，如果已经存在缓存文件中则直接读取
        # embedding_path = os.path.join(self.cache_dir, f"{hash(self.image_dir + '_' + self.image_files[idx])}.pkl")
        # if os.path.exists(embedding_path):
        #     with open(embedding_path, 'rb') as f:
        #         image_embedding = pickle.load(f)
        #     self.autoMask.set_cached_embedding(image_embedding=image_embedding,shape=image.shape[:2])

        # masks = self.autoMask.generate(image=image)
        # image_embedding = self.autoMask.predictor.get_image_embedding().to(self.device)
        # with open(embedding_path, 'wb') as f:
        #     pickle.dump(image_embedding, f)
        # self.autoMask.predictor.reset_image()

        # # 随机选择一个mask并进行扭曲
        # mask = np.zeros_like(image[:,:,0], dtype=np.float32)
        # for i, m in enumerate(masks[::-1]):
        #     mask[m['segmentation']] = i + 1

        # 单应性变换
        data_dict = generateTrainImagePair(image, image, patch_size=self.patch_size)

        # 将图像和扭曲后的mask转换为tensor
        data_dict['img1_patch'] = torch.from_numpy(data_dict['img1_patch']).permute(2, 0, 1).float() / 255.0
        data_dict['img2_patch'] = self._process_mask(torch.from_numpy(data_dict['img2_patch'])) / 255.0
        data_dict['img1_full'] = torch.from_numpy(data_dict['img1_full']).permute(2, 0, 1) / 255.0
        data_dict['img2_full'] = self._process_mask(torch.from_numpy(data_dict['img2_full'])) / 255.0
        data_dict['disturbed_corners'] = torch.from_numpy(data_dict['disturbed_corners']).permute(2, 0, 1).float()

        # 如果有其他transform，进行处理
        # if self.transform:
        #     image = self.transform(image)

        return data_dict


# 示例用法
# from segment_anything import sam_model_registry, SamPredictor
# sam = sam_model_registry['vit_h'](checkpoint="path/to/sam/checkpoint")
# predictor = SamPredictor(sam)
# dataset = SAMAugmentedDataset(image_dir="path/to/images", predictor=predictor)
# (image, warped_mask), flow = dataset[0]
