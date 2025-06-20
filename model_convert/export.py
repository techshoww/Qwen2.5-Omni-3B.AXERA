import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
from transformers import  AutoTokenizer, AutoProcessor
from modeling_export import Qwen2_5OmniModel_Export
from qwen_vl_utils import process_vision_info
import numpy as np 
import os
import sys 

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
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=onnx_output+".data"
                )
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))

def generate_attnmask(seq_length, cu_seqlens, device):
    attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

    return attention_mask

checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen2.5-Omni-3B-Instruct/"
# which = sys.argv[2] if len(sys.argv)>=3 else "image"
which = "video"

# default: Load the model on the available device(s)
model = Qwen2_5OmniModel_Export.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cpu"
)

export_model = model.thinker.visual

if which=="image":
    export_model.forward = export_model.forward_export
elif which=="video":
    export_model.forward = export_model.forward_export_by_second_nchw
device = torch.device("cpu")


hidden_states = torch.load("hidden_states.pth",weights_only=True).to(torch.float32).to(device)
print("hidden_states",hidden_states.shape)

input = ( hidden_states)

input_names = ["hidden_states"]

onnx_output = f"Qwen2.5-Omni-3B_vision.onnx"

output_names = [f"hidden_states_out"]


export_onnx(export_model, input, input_names, output_names, onnx_output)    

