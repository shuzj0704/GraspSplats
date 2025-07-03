# put this file in the folder of weights

import os
import torch
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor

# 定义本地权重文件路径和映射关系
WEIGHTS_DIR = "/home/shu/3DGS/GraspSplats/weights"
ENCODER_MAPPING = {
    'efficientvit_l2': 'l2.pt',
    'tiny_vit': 'mobile_sam.pt',
    'sam_vit_h': 'sam_vit_h.pt'
}

def _get_object_aware_model():
    """加载目标感知模型"""
    # 本地文件路径
    object_aware_model_path = os.path.join(WEIGHTS_DIR, 'ObjectAwareModel.pt')
    assert os.path.exists(object_aware_model_path), f"权重文件 {object_aware_model_path} 不存在"
    
    # 设置模型
    ObjAwareModel = ObjectAwareModel(object_aware_model_path)
    return ObjAwareModel

def _get_mobilesamv2(encoder_type):
    """加载MobileSAMv2主模型"""
    # 获取编码器对应的文件名
    encoder_filename = ENCODER_MAPPING.get(encoder_type)
    if encoder_filename is None:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
    # 本地文件路径
    encoder_path = os.path.join(WEIGHTS_DIR, encoder_filename)
    decoder_path = os.path.join(WEIGHTS_DIR, 'Prompt_guided_Mask_Decoder.pt')
    
    # 检查文件是否存在
    assert os.path.exists(encoder_path), f"编码器权重文件 {encoder_path} 不存在"
    assert os.path.exists(decoder_path), f"解码器权重文件 {decoder_path} 不存在"

    # 设置模型
    PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](decoder_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
    image_encoder = sam_model_registry[encoder_type](encoder_path)
    mobilesamv2.image_encoder = image_encoder
    mobilesamv2.eval()

    return mobilesamv2

def _get_everything(encoder_type):
    """获取完整模型组件"""
    obj_aware_model = _get_object_aware_model()
    mobilesamv2 = _get_mobilesamv2(encoder_type)
    predictor = SamPredictor(mobilesamv2)
    return mobilesamv2, obj_aware_model, predictor

def mobilesamv2_efficientvit_l2(pretrained=True, **kwargs):
    """使用efficientvit_l2编码器的MobileSAMv2模型"""
    assert pretrained, "仅支持推理模式，不支持训练"
    return _get_everything('efficientvit_l2')

def mobilesamv2_tiny_vit(pretrained=True, **kwargs):
    """使用tiny_vit编码器的MobileSAMv2模型"""
    assert pretrained, "仅支持推理模式，不支持训练"
    return _get_everything('tiny_vit')

def mobilesamv2_sam_vit_h(pretrained=True, **kwargs):
    """使用sam_vit_h编码器的MobileSAMv2模型"""
    assert pretrained, "仅支持推理模式，不支持训练"
    return _get_everything('sam_vit_h')