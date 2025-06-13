
import sys
import os
from qwen_omni_utils import process_mm_info
import librosa
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
from onnx import helper
from io import BytesIO
from urllib.request import urlopen
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from modeling_export import Qwen2_5OmniModel_Export
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
    # [libprotobuf ERROR /tmp/pip-install-g8oov4oc/onnxsim_7f61fd2e2c6d42a2abf5f2968fd227a6/third_party/onnx-optimizer/third_party/protobuf/src/google/protobuf/message_lite.cc:449] onnx.ModelProto exceeded maximum protobuf size of 2GB: 2878887489

device = torch.device("cpu")
model_path = "/data/lihongjie/Qwen2.5-Omni-3B"
model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=device,
)


norm_model = model.thinker.model.norm
lm_head = model.thinker.lm_head


x = torch.ones([1,336,2048], dtype=torch.float32).to(device)
input = (x,)
input_names=["input"]
export_onnx(norm_model, input, input_names=input_names, output_names=["output"], onnx_output="norm_prefill_352.onnx")
export_onnx(lm_head, input, input_names=input_names, output_names=["output"], onnx_output="lm_head_prefill_352.onnx")

x = torch.ones([1,1,2048], dtype=torch.float32).to(device)
input = (x,)
input_names=["input"]
export_onnx(norm_model, input, input_names=input_names, output_names=["output"], onnx_output="norm_decode.onnx")
export_onnx(lm_head, input, input_names=input_names, output_names=["output"], onnx_output="lm_head_decode.onnx")

