
from dataset import BaseDataset,SAMAugmentedDataset
from segment_anything import sam_model_registry, SamPredictor
import math
import os
import argparse

SAMPATH="/CV/gaobiaoli/project/weights/sam_vit_b_01ec64.pth"
DEFAULT="/CV/gaobiaoli/dataset/CIS-Dataset/train"
CACHE="/CV/gaobiaoli/cache"
# SAMPATH="/home/gaobiaoli/weights/sam_vit_b_01ec64.pth"
# DEFAULT="/CV/3T/dataset-public/COCO/coco2017/train2017"
# CACHE="/CV/3T/dataset-public/COCO/coco2017/cache"
def split_data(image_files, num_splits, process_id):
    """
    使用模运算分配数据集，确保不同进程负责不同部分
    """
    # subset = [image_files[i] for i in range(len(image_files)) if i % num_splits == process_id]
    subset = image_files[::-1]
    print(subset[0:10])
    return subset

def create_sam_predictor_for_device(args):
    """
    根据设备 ID 和分配的文件列表创建 SAM 并生成缓存
    """
    # 加载 SAM 模型
    sam_model_path = SAMPATH
    sam = sam_model_registry['vit_b'](checkpoint=sam_model_path).to(device=f"cuda:{args.process_id}")
    
    image_files = os.listdir(args.image_dir)

    # 根据进程编号进行数据分割
    image_files_subset = split_data(image_files, args.num_processes, args.process_id)


    # 创建 SAM 数据集
    dataset = SAMAugmentedDataset(
        sam=sam,
        image_dir = args.image_dir,
        image_files = image_files_subset,
        cache_dir=CACHE,
        rho=16,
        ratio=0.01,
        patch_size=512,
        reshape=(1024, 1024),
        keep_ratio=False
    )
    
    # 生成缓存
    dataset.generateSAMCache()
if __name__ =="__main__":

    parser = argparse.ArgumentParser(description="生成 SAM 缓存的多进程工具")
    parser.add_argument(
        "--process_id", 
        type=int, 
        required=True, 
        choices=[0, 1, 2, 3], 
        help="进程 ID，选择 0, 1, 2, 3 之一"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=4, 
        help="总进程数，默认为 4"
    )
    parser.add_argument(
            "--image_dir", 
            type=str, 
            default=DEFAULT, 
            help="图像数据集目录"
        )
    args = parser.parse_args()

    create_sam_predictor_for_device(args)
