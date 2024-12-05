from PIL import Image
import torch
import numpy as np
import matplotlib.cm as cm
import random
def show_tensor(tensor):
    '''
    可视化tensor:转换通道 + 转为np.uint8
    '''
    if tensor.is_cuda:
        tensor = tensor.cpu()
    ndarray = tensor.permute(1, 2, 0).numpy()  # 调整维度为 (W, H, C)
    ndarray = np.clip(ndarray, 0, 255).astype(np.uint8)
    return Image.fromarray(ndarray)

def display_images_in_row(images, padding=10, bg_color=(255, 255, 255)):
    """
    将多个图像在一行中拼接显示。

    Args:
        images (list): 图像列表，每个图像为 PIL.Image.Image 实例，或者可以转换为图像的 numpy 数组。
        padding (int): 图像之间的间距。
        bg_color (tuple): 背景颜色 (R, G, B)。
    
    Returns:
        PIL.Image.Image: 拼接后的图像。
    """
    for i in range(len(images)):
        if not isinstance(images[i],Image.Image):
            images[i]=Image.fromarray(images[i])

    # 获取每个图像的宽高，并计算总宽度和最大高度
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    # 创建背景图像
    combined_image = Image.new("RGB", (total_width, max_height), bg_color)
    
    # 拼接图像
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, (max_height - img.height) // 2))
        x_offset += img.width + padding

    return combined_image



def generate_unique_colors(num_colors):
    """生成不重复的随机颜色"""
    random.seed(0)
    colors = []
    for _ in range(num_colors):
        color = [random.randint(0, 255) for _ in range(3)]  # 随机RGB
        colors.append(color)
    return np.array(colors, dtype=np.uint8)

def show_mask(mask):
    # 如果是多通道的 mask，压缩成单通道
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:  # (1, H, W) -> (H, W)
            mask = mask[0]
        elif mask.shape[-1] == 1:  # (H, W, 1) -> (H, W)
            mask = mask[:, :, 0]

    # 获取唯一标签
    unique_labels = np.unique(mask)
    
    # 创建一个空的彩色掩膜
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # 生成随机颜色
    num_colors = len(unique_labels) - 1  # 排除背景标签 0
    colors = generate_unique_colors(num_colors)

    # 给每个标签分配颜色
    for i, label in enumerate(unique_labels):
        if label > 0:  # 忽略背景标签（假设背景标签为0）
            colored_mask[mask == label] = colors[i - 1]  # 将颜色分配给标签

    return colored_mask