# scripts/verify_downloads.py
# Run this FIRST. If it fails, fix before proceeding.
import torch
from transformers import AutoModel
from timm import create_model
from huggingface_hub import hf_hub_download
import os

def check(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

check("ResNet50",        lambda: create_model('resnet50', pretrained=True))
check("EfficientNet-B3", lambda: create_model('efficientnet_b3', pretrained=True))
check("Swin-T",          lambda: create_model('swin_tiny_patch4_window7_224', pretrained=True))
check("ViT-B/16",        lambda: create_model('vit_base_patch16_224', pretrained=True))
check("Phikon-v2",       lambda: AutoModel.from_pretrained("owkin/phikon-v2"))
check("H-Optimus-0",     lambda: AutoModel.from_pretrained("bioptimus/H-optimus-0", trust_remote_code=True))
check("Prov-GigaPath",   lambda: create_model("hf-hub:prov-gigapath/prov-gigapath", pretrained=True))
