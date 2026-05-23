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
 

class OmniPerceiveExportWrapper(torch.nn.Module):
    """
    Thin wrapper that unwraps the output dicts into flat named tensors
    so ONNX export produces clean named outputs (no Python dicts).
    """
    def __init__(self, model: OmniPerceive):
        super().__init__()
        self.model = model
 
    def forward(self, image: torch.Tensor):
        out = self.model(image)        # inference mode — returns dict
 
        det   = out["det"]
        lane  = out["lane"]
        depth = out["depth"]
        seg   = out["seg"]
 
        return (
            det["heatmap"],           # det_heatmap
            det["offset"],            # det_offset
            det["wh"],                # det_wh
            lane["lane_coeffs"],      # lane_coeffs
            lane["lane_conf"],        # lane_conf
            depth["depth"],           # depth_map
            seg,                      # seg_logits
        )
 
 
def export(args):
    logger = get_logger("export_onnx", "runs/")
 
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
 
    device = torch.device("cpu")    # always export on CPU for portability
    model  = OmniPerceive(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, logger=logger)
 
    wrapper = OmniPerceiveExportWrapper(model).eval()
 
    dummy = torch.randn(args.batch_size, 3, args.input_h, args.input_w)
 
    # Dry run to check shapes
    with torch.no_grad():
        outputs = wrapper(dummy)
    output_names = [
        "det_heatmap", "det_offset", "det_wh",
        "lane_coeffs", "lane_conf",
        "depth_map", "seg_logits",
    ]
    logger.info("PyTorch output shapes:")
    for name, t in zip(output_names, outputs):
        logger.info(f"  {name:20s}: {tuple(t.shape)}")
 
    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting ONNX (opset {args.opset}) → {args.output}")
 
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes={
            "image":        {0: "batch_size"},
            "det_heatmap":  {0: "batch_size"},
            "depth_map":    {0: "batch_size"},
            "seg_logits":   {0: "batch_size"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    logger.info(f"Export complete: {args.output}")
 
    # Optional: verify ONNX Runtime outputs match PyTorch
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
 
            sess = ort.InferenceSession(
                args.output,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            ort_inputs  = {"image": dummy.numpy()}
            ort_outputs = sess.run(None, ort_inputs)
 
            logger.info("ONNX Runtime vs PyTorch verification:")
            all_ok = True
            for name, pt_out, ort_out in zip(output_names, outputs, ort_outputs):
                pt_np   = pt_out.numpy()
                max_err = np.abs(pt_np - ort_out).max()
                rel_err = max_err / (np.abs(pt_np).max() + 1e-8)
                ok = rel_err < 1e-4
                all_ok &= ok
                logger.info(f"  {name:20s}: max_err={max_err:.2e}  rel_err={rel_err:.2e}  {'✅' if ok else '❌'}")
 
            if all_ok:
                logger.info("All outputs match within tolerance ✅")
            else:
                logger.warning("Some outputs exceed tolerance — check ONNX graph ⚠️")
 
        except ImportError:
            logger.warning("onnxruntime not installed — skipping verification.")
            logger.warning("Install: pip install onnxruntime-gpu")
 
