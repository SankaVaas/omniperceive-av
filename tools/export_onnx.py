import argparse
import torch
import yaml
from pathlib import Path
 
from models.omniperceive import OmniPerceive
from utils.checkpoint    import load_checkpoint
from utils.logger        import get_logger
 
 
def parse_args():
    p = argparse.ArgumentParser("OmniPerceive ONNX Export")
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output",      default="weights/omniperceive.onnx")
    p.add_argument("--input_h",     type=int, default=384)
    p.add_argument("--input_w",     type=int, default=1280)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--verify",      action="store_true",
                   help="Run ONNX Runtime and verify outputs match PyTorch")
    p.add_argument("--opset",       type=int, default=17)
    return p.parse_args()
 