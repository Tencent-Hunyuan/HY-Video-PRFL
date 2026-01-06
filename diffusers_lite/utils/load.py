import torch
from peft import PeftModel
from diffusers_lite.wan.modules.model import WanModel, WanAttentionBlock


def get_no_split_modules(transformer):
    while isinstance(transformer, PeftModel):
        transformer = transformer.base_model.model
    if isinstance(transformer, WanModel):
        return (WanAttentionBlock, )
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")