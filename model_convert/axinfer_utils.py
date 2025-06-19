import os

import numpy as np
import onnxruntime as ort
from axengine import InferenceSession
from ml_dtypes import bfloat16
# from scipy.special import softmax
from transformers import AutoTokenizer
import gc
import dill 
import torch
from torch import nn

class InferEngine:
    def __init__(self, path):
        self.path = path

        model_type = os.path.splitext(path)[1]

        engine = None
        if model_type == ".onnx":
            engine = ort.InferenceSession
        elif model_type == ".axmodel":
            engine = InferenceSession
        else:
            raise NotImplementedError(f"Not supported model type: {model_type}")

        assert engine is not None

        self.session = engine(path)

    def run(self, outputnames, inputs, shape_group=None):
        if shape_group is None:
            outputs = self.session.run(outputnames, inputs)
        else:
            outputs = self.session.run(outputnames, inputs, shape_group=shape_group)
        return outputs


class AxModelInferStatic:
    def __init__(self, axmodel_path):
        self.session = InferEngine(axmodel_path)

    def __call__(self, inputs, shape_group=None):

        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs


class AxModelInferDynamic:
    def __init__(self, axmodel_path):
        self.axmodel_path = axmodel_path

    def __call__(self, inputs, shape_group=None):

        session = InferEngine(self.axmodel_path)
        if shape_group is None:
            outputs = session.run(None, inputs)
        else:
            outputs = session.run(None, inputs, shape_group=shape_group)

        return outputs


class AxModelInfer:
    def __init__(self, axmodel_path, run_dynamic=False):
        if run_dynamic:
            self.model = AxModelInferDynamic(axmodel_path)
        else:
            self.model = AxModelInferStatic(axmodel_path)

    def __call__(self, inputs, shape_group=None):
        try:
            outputs = self.model(inputs, shape_group)
        except Exception as e:
            if hasattr(self.model, "axmodel_path"):
                print(f"axmodel_path:{self.model.axmodel_path}")
            print(e)
            raise e

        return outputs

class NormLayer:
    def __init__(self,  axmodel_path, prefill_len,  run_dynamic=False, data_type=bfloat16):
        self.prefill_len = prefill_len
        self.data_type = data_type
        self.model = AxModelInfer(axmodel_path, run_dynamic=run_dynamic)
       
    def __call__(self, hidden_states, is_prefill):
        hidden_states = hidden_states.cpu().numpy().astype(self.data_type)
        if is_prefill:
            token_len = hidden_states.shape[1]
            x = np.zeros((hidden_states.shape[0], self.prefill_len, hidden_states.shape[2]) , dtype=self.data_type)
            x[:, 0:token_len] = hidden_states
        else:
            x = hidden_states
        input_feed = {
            "input": x
        }

        if is_prefill:
            output = self.model(input_feed, shape_group=1)[0][:,0:token_len]
        else:
            output = self.model(input_feed, shape_group=0)[0]

        output = torch.from_numpy(output.astype(np.float32))
        return output

class LMHead:
    def __init__(self,  axmodel_path, prefill_len,  run_dynamic=False, data_type=bfloat16):
        self.prefill_len = prefill_len
        self.data_type = data_type
        self.model = AxModelInfer(axmodel_path, run_dynamic=run_dynamic)
       
    def __call__(self, hidden_states, is_prefill):
        hidden_states = hidden_states.cpu().numpy().astype(self.data_type)
        if is_prefill:
            token_len = hidden_states.shape[1]
            x = np.zeros((hidden_states.shape[0], self.prefill_len, hidden_states.shape[2]) , dtype=self.data_type)
            x[:, 0:token_len] = hidden_states
        else:
            x = hidden_states
        input_feed = {
            "input": x
        }

        if is_prefill:
            output = self.model(input_feed, shape_group=1)[1][:,0:token_len]
        else:
            output = self.model(input_feed, shape_group=0)[1]

        output = torch.from_numpy(output.astype(np.float32))
        return output
        
class PostLayer:
    def __init__(self,  axmodel_path_prefill, axmodel_path_decode, prefill_len,  run_dynamic=False, data_type=bfloat16):
        self.prefill_len = prefill_len
        self.data_type = data_type
        self.model_prefill = AxModelInfer(axmodel_path_prefill, run_dynamic=run_dynamic)
        self.model_decode = AxModelInfer(axmodel_path_decode, run_dynamic=run_dynamic)


    def __call__(self, hidden_states, is_prefill):
        hidden_states = hidden_states.cpu().numpy().astype(self.data_type)
        if is_prefill:
            token_len = hidden_states.shape[1]
            x = np.zeros((hidden_states.shape[0], self.prefill_len, hidden_states.shape[2]) , dtype=self.data_type)
            x[:, 0:token_len] = hidden_states
        else:
            x = hidden_states
        input_feed = {
            "input": x
        }

        if is_prefill:
            output = self.model_prefill(input_feed)[0][:,0:token_len]
        else:
            output = self.model_decode(input_feed)[0]

        output = torch.from_numpy(output.astype(np.float32))
        return output

class LMLayer:
    def __init__(self, config, axmodel_path, prefill_len, lastN,  run_dynamic=False, data_type=bfloat16):
        self.prefill_len = prefill_len
        self.lastN = lastN
        self.hidden_size = config.hidden_size
        self.data_type = data_type
        kv_dim = (
            config.hidden_size
            // config.num_attention_heads
            * config.num_key_value_heads
        )

        self.k_cache = np.zeros((1, self.lastN, kv_dim), dtype=data_type)
        self.v_cache = np.zeros((1, self.lastN, kv_dim), dtype=data_type)
        self.cache_idx = 0
        self.model = AxModelInfer(axmodel_path, run_dynamic)

        self.mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        self.mask[:, :, : self.lastN] -= 65536

    def __call__(self, input_embeds,  position_ids,  is_prefill=None):

        assert is_prefill is not None 

        if is_prefill:
            data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(self.data_type)
            token_len = input_embeds.shape[1]
            data[:, 0:token_len] = input_embeds.cpu().numpy().astype(self.data_type)

            mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
            # mask[:, 0:token_len, 0:token_len] = attention_mask
            for i in range(token_len):
                mask[:, i, : i + 1] = 0
            mask = mask.astype(bfloat16)

            past_k = np.zeros((1, 1, self.hidden_size), dtype=self.data_type) if is_prefill else self.k_cache
            past_v = np.zeros((1, 1, self.hidden_size), dtype=self.data_type) if is_prefill else self.v_cache

            indices = np.zeros((3, self.prefill_len), dtype=np.uint32)
            indices[:, 0:token_len] = position_ids.squeeze(1).cpu().numpy().astype(np.uint32)
        else:
            # print("decode self.cache_idx(start_indice)",self.cache_idx)
            data = input_embeds.cpu().numpy().astype(self.data_type)
           
            mask = self.mask

            past_k = self.k_cache
            past_v = self.v_cache
            assert torch.equal(position_ids[0], position_ids[1]) and torch.equal(position_ids[0], position_ids[2])
            indices = position_ids[0:1].squeeze(1).cpu().numpy().astype(np.uint32)

        input_feed = {
            "K_cache": past_k,
            "V_cache": past_v,
            "indices": indices,
            "input": data,
            "mask": mask
        }

        shape_group = 1 if is_prefill else 0
        outputs = self.model(input_feed, shape_group=shape_group)

        if is_prefill:
            hidden_states = outputs[2][:, 0:token_len]
            self.k_cache[:,0:token_len] = outputs[0][:, 0:token_len]
            self.v_cache[:,0:token_len] = outputs[1][:, 0:token_len]
            self.cache_idx = token_len 
            self.mask[:, :, :token_len] = 0
        else:
            hidden_states = outputs[2]
            self.k_cache[:, self.cache_idx] = outputs[0]
            self.v_cache[:, self.cache_idx] = outputs[1]
            self.mask[:, :, :self.cache_idx ] = 0

            self.cache_idx += 1

        return torch.from_numpy(hidden_states.astype(np.float32))