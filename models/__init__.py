from .icrnet_v2_psp import ICRNet_V2PSP

def get_model(version = '2.1', **kwargs):
    return ICRNet_V2PSP(**kwargs) # 语义分割 + 合理性识别，基于PSPNet与DenseNet
