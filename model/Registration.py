import torch.nn as nn
import torch

class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class VGGRegistrationModel(nn.Module):
    def __init__(self, patch_size=224, batch_norm=True):
        super(VGGRegistrationModel, self).__init__()
        
        # 定义CNN部分（特征提取）
        self.cnn1 = nn.Sequential(
            # Block(3, 64, batch_norm),  # 输入是RGB图像
            # Block(64, 128, batch_norm),
            # Block(128, 256, batch_norm),
            Block(512, 512, batch_norm, pool=False),  # 不进行池化，保持空间分辨率
        )
        
        # 定义全连接层部分（输出四个点的偏移）
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512 * (patch_size // 16) * (patch_size // 16), 1024),  # 特征图展平
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),  # 输出8个参数，表示四个点的偏移
        )

    def forward(self, img1_feas, img2_feas):
        # 从两个图像特征列表中取最后一个特征图
        last_fea_1 = img1_feas[-2] # 获取img_fea_list_1中的最后一层特征
        last_fea_2 = img2_feas[-2]  # 获取img_fea_list_2中的最后一层特征

        # 将最后一层特征图拼接在一起
        combined_features = torch.cat((last_fea_1, last_fea_2), dim=1)

        # 通过卷积提取特征
        x = self.cnn1(combined_features)
        
        # 通过全连接层进行回归预测
        output = self.fc(x)
        return output