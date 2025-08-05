
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

device = torch.device("cuda:1")
model_path = sys.argv[1]
model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=device,
)


model = model.talker.thinker_to_talker_proj

prefill_len=int(sys.argv[2])
x = torch.ones([1,prefill_len,2048], dtype=torch.float32).to(device)
# y = torch.ones([1,1,896], dtype=torch.float32).to(device)
input = (x,)
input_names=["input"]
export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output=f"thinker_to_talker_proj_prefill_{prefill_len}.onnx")

x = torch.ones([1,1,2048], dtype=torch.float32).to(device)
# y = torch.ones([1,1,896], dtype=torch.float32).to(device)
input = (x,)
input_names=["input"]
export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output="thinker_to_talker_proj_decode.onnx")
