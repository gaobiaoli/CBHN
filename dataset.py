import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from SAM.SamAutomaticMaskGenerator import SamAutomaticMaskGenerator
from utils.transform_utils import generateTrainImagePair
import pickle
import random
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(
        self,
        image_dir,
        cache_dir="cache",
        device="cuda",
        transform=None,
        ratio=1.0,
        rho=16,
        reshape=(640, 640),
        keep_ratio=True,
        patch_size=328,
    ):
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_files = self.image_files[0 : int(len(self.image_files) * ratio)]
        self.device = device
        self.transform = transform
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.patch_size = patch_size
        self.reshape = reshape
        self.keep_ratio = keep_ratio
        self.rho = rho
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.cache_prob = 0.0

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

        cache_file = os.path.join(self.cache_dir, f"{self.image_files[idx]}.pkl")

        if random.random() < self.cache_prob and os.path.exists(cache_file):
            # 读取缓存
            with open(cache_file, "rb") as f:
                data_dict = pickle.load(f)
        else:
            # 重新生成数据
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            data_dict = self._generate_data(image, image)

            # 将生成的数据保存到缓存
            # with open(cache_file, "wb") as f:
            #     pickle.dump(data_dict, f)

        # 如果有其他transform，进行处理
        # if self.transform:
        #     image = self.transform(image)

        return data_dict

    def _generate_data(self, img1, img2):
        # 单应性变换或其他数据生成过程
        data_dict = generateTrainImagePair(
            img1,
            img2,
            marginal=self.rho,
            patch_size=self.patch_size,
            reshape=self.reshape,
            keep_ratio=self.keep_ratio,
        )

        # 将图像和扭曲后的mask转换为tensor
        data_dict["img1_patch"] = (
            torch.from_numpy(self.normlize(data_dict["img1_patch"]))
            .permute(2, 0, 1)
            .float()
        )
        data_dict["img2_patch"] = self._process_mask(
            torch.from_numpy(self.normlize(data_dict["img2_patch"]))
        ).float()
        data_dict["img1_full"] = (
            torch.from_numpy(self.normlize(data_dict["img1_full"]))
            .permute(2, 0, 1)
            .float()
        )
        data_dict["img2_full"] = self._process_mask(
            torch.from_numpy(self.normlize(data_dict["img2_full"]))
        ).float()
        data_dict["disturbed_corners"] = (
            torch.from_numpy(data_dict["disturbed_corners"]).permute(2, 0, 1).float()
        )

        return data_dict

    def generateCache(self):
        # 遍历所有图像并生成缓存
        for idx in range(len(self.image_files)):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            data_dict = self._generate_data(image)

            # 构建缓存文件路径
            cache_file = os.path.join(self.cache_dir, f"{self.image_files[idx]}.pkl")

            # 保存到缓存
            with open(cache_file, "wb") as f:
                pickle.dump(data_dict, f)

    def normlize(self, ndarray):
        if len(ndarray.shape) == 2 or ndarray.shape[2] != 3:  # (w,h,1)
            return ndarray / 255.0  # 单通道图像归一化

        return (ndarray - self.mean_I) / self.std_I

    def denormlize(self, ndarray):
        if len(ndarray.shape) == 2 or ndarray.shape[2] != 3:  # (w,h,1)
            return ndarray * 255.0  # 单通道图像归一化

        return (ndarray * self.std_I) + self.mean_I


class SAMAugmentedDataset(BaseDataset):
    def __init__(
        self,
        sam,
        *args,
        **kwargs,
    ):
        # 初始化图像文件夹路径，SAM模型，设备等参数
        super().__init__(*args, **kwargs)
        # 初始化SAM相关参数
        self.autoMask = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.2,
            stability_score_thresh=0.8,
            stability_score_offset=2,
            crop_nms_thresh=0.5,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        cache_file = os.path.join(self.cache_dir, f"{self.image_files[idx]}.pkl")

        if random.random() < 1.0 and os.path.exists(cache_file):
            # 读取缓存
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            img1 = (
                cache["image"]
                if "image" in cache.keys()
                else cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            )
            img2 = cache["mask"]
            data_dict = self._generate_data(img1=img1, img2=img2)
        else:
            # 重新生成数据
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # embedding_path = os.path.join(
            #     self.cache_dir,
            #     f"{hash(self.image_dir + '_' + self.image_files[idx])}.pkl",
            # )
            # if os.path.exists(embedding_path):
            #     with open(embedding_path, "rb") as f:
            #         image_embedding = pickle.load(f)
            #     self.autoMask.set_cached_embedding(
            #         image_embedding=image_embedding, shape=image.shape[:2]
            #     )
            masks = self.autoMask.generate(image=image)
            # image_embedding = self.autoMask.predictor.get_image_embedding().to(
            #     self.device
            # )
            # with open(embedding_path, "wb") as f:
            #     pickle.dump(image_embedding, f)
            self.autoMask.predictor.reset_image()
            mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
            for i, m in enumerate(masks[::-1]):
                mask[m["segmentation"]] = i + 1

            data_dict = self._generate_data(img1=image, img2=mask)
        return data_dict
        # return image,mask

    def generateCache(self):
        # 遍历所有图像并生成缓存
        for idx in tqdm(range(len(self.image_files))):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            masks = self.autoMask.generate(image=image)
            # image_embedding = self.autoMask.predictor.get_image_embedding().to(self.device)
            self.autoMask.predictor.reset_image()
            mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
            for i, m in enumerate(masks[::-1]):
                mask[m["segmentation"]] = i + 1

            data_dict = self._generate_data(image, mask)

            # data_dict["img1_full"]=1
            # data_dict["img2_full"]=1
            # 构建缓存文件路径
            cache_file = os.path.join(self.cache_dir, f"{self.image_files[idx]}.pkl")

            # 保存到缓存
            with open(cache_file, "wb") as f:
                pickle.dump(data_dict, f)

    def generateSAMCache(self):
        # 遍历所有图像并生成缓存
        for idx in tqdm(range(len(self.image_files))):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            masks = self.autoMask.generate(image=image)
            # image_embedding = self.autoMask.predictor.get_image_embedding().to(self.device)
            self.autoMask.predictor.reset_image()
            mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
            for i, m in enumerate(masks[::-1]):
                mask[m["segmentation"]] = i + 1

            cache_file = os.path.join(self.cache_dir, f"{self.image_files[idx]}.pkl")
            data_dict = {"image": image, "mask": mask}
            # 保存到缓存
            with open(cache_file, "wb") as f:
                pickle.dump(data_dict, f)
