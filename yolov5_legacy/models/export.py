"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats.

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

# 首先pip install onnx
import argparse
import sys
import time

sys.path.append("./")  # to run '$ python *.py' files in subdirectories
sys.path.append("../")
import models
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import check_img_size, set_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./yolov5s.pt", help="weights path")  # from yolov5/models/
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640], help="image size")  # height, width
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device("cpu"))  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    try:
        print(f"\nStarting TorchScript export with torch {torch.__version__}...")
        f = opt.weights.replace(".pt", ".torchscript.pt")  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print(f"TorchScript export success, saved as {f}")
    except Exception as e:
        print(f"TorchScript export failure: {e}")

    # ONNX export
    try:
        import onnx

        print(f"\nStarting ONNX export with onnx {onnx.__version__}...")
        f = opt.weights.replace(".pt", ".onnx")  # filename
        torch.onnx.export(
            model,
            img,
            f,
            verbose=False,
            opset_version=12,
            input_names=["images"],
            output_names=["classes", "boxes"] if y is None else ["output"],
        )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print(f"ONNX export success, saved as {f}")
    except Exception as e:
        print(f"ONNX export failure: {e}")

    # CoreML export
    try:
        import coremltools as ct

        print(f"\nStarting CoreML export with coremltools {ct.__version__}...")
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name="image", shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace(".pt", ".mlmodel")  # filename
        model.save(f)
        print(f"CoreML export success, saved as {f}")
    except Exception as e:
        print(f"CoreML export failure: {e}")

    # Finish
    print("\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron." % (time.time() - t))
