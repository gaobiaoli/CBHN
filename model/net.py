from .BaseModel import BaseModel
from torchvision.models import resnet34,ResNet
from .resnet import resnet34,simplenet
from .Registration import VGGRegistrationModel
from .swin_multi import SwinTransformer
from .swin import SingleSwinTransformer
def getDemoModel(img2_channel=3,param=None):
    backbone1=simplenet(input_channels=3,param=param)
    # if img2_channel==3:
    #     backbone2=backbone1
    # else:
    #     backbone2=resnet34(in_channels=img2_channel)
    regis=SingleSwinTransformer(param=param)
    model=BaseModel(backbone_1=backbone1,backbone_2=backbone1,registration=regis)
    return model