
import sys
import os
from qwen_omni_utils import process_mm_info
import librosa
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnxslim
from onnx import helper
from io import BytesIO
from urllib.request import urlopen
import torch
from audio_export import Qwen2_5OmniModel_Export
# @title inference function
def export_onnx(model, input, input_names, output_names, onnx_output):

    torch.onnx.export(
        model,
        input,
        onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )

    onnx_model = onnx.load(onnx_output)
    print("IR 版本:", onnx_model.ir_version)
    print("操作集:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    # onnx.save(onnx_model, onnx_output)
    # convert model
    
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_output,
                # save_as_external_data=True,
                # all_tensors_to_one_file=True,
                # location=onnx_output+".data"
                )
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))
   
    # os.system(f"onnxslim {onnx_output} {onnx_output} ")


device = torch.device("cpu")
model_path = "/data/lihongjie/Qwen2.5-Omni-3B"
model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=device,
)


model = model.thinker.audio_tower
model.forward = model.forward_export


padded_feature = torch.load("padded_feature.pth")
padded_mask = torch.load("padded_mask.pth")
attention_mask = torch.load("attention_mask.pth")
input = (padded_feature, padded_mask,  attention_mask)
input_names=["padded_feature", "padded_mask",  "attention_mask"]
export_onnx(model, input, input_names=input_names, output_names=["token_audio"], onnx_output="audio_tower.onnx")
