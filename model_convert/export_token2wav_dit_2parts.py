
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
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
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

device = torch.device("cuda:5")
model_path = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)


model = model.token2wav.code2wav_dit_model
model.part_num = 8

# export part1
model.forward = model.forward_part1

x = torch.load("x.pth").to(model.device)
cond = torch.load("cond.pth").to(model.device)
spk = torch.load("spk.pth").to(model.device)
code = torch.load("code.pth").to(model.device)
time = torch.load("time.pth").to(model.device)
input = (x, cond, spk, code, time)
input_names=["x", "cond", "spk", "code", "time"]
export_onnx(model, input, input_names=input_names, output_names=["hidden", "t"], onnx_output="token2wav_dit_part1.onnx")

for i in range(1, model.part_num):

    model.forward = model.forward_part2
    model.part_idx = i

    hidden = torch.load("hidden_part2.pth").to(model.device)
    t = torch.load("t_part2.pth").to(model.device)

    input = (hidden, t)
    input_names = ["hidden", "t"]
    export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output=f"token2wav_dit_part{i+1}.onnx")

# # export part2
# model.forward = model.forward_part2

# hidden = torch.load("hidden_part2.pth").to(model.device)
# t = torch.load("t_part2.pth").to(model.device)

# input = (hidden, t)
# input_names = ["hidden", "t"]
# export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output="token2wav_dit_part2.onnx")


# # export part3
# model.forward = model.forward_part3
# hidden = torch.load("hidden_part2.pth").to(model.device)
# t = torch.load("t_part2.pth").to(model.device)

# input = (hidden, t)
# input_names = ["hidden", "t"]
# export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output="token2wav_dit_part3.onnx")


# # export part4
# model.forward = model.forward_part4
# hidden = torch.load("hidden_part2.pth").to(model.device)
# t = torch.load("t_part2.pth").to(model.device)

# input = (hidden, t)
# input_names = ["hidden", "t"]
# export_onnx(model, input, input_names=input_names, output_names=["output"], onnx_output="token2wav_dit_part4.onnx")