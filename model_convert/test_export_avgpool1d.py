import torch 
import torch.nn as nn 
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnxslim
from onnx import helper

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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        return self.avg_pooler(x)


model = Model()

input = torch.ones((3584,100))

export_onnx(model, input, input_names=["input"], output_names=["output"], onnx_output="test.onnx")