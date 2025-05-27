
import torch 
from torch import nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import onnxruntime as ort
from qwen_omni_utils import process_mm_info
from transformers import AutoTokenizer, AutoConfig, Qwen2_5OmniProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder, Qwen2_5OmniThinkerConfig,\
Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionSdpaAttention, apply_rotary_pos_emb_vision, Qwen2_5OmniVisionAttention, Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel,\
    kaiser_sinc_filter1d
from audio_export import Qwen2_5OmniAudioEncoder_Export
from token2wav_export import Qwen2_5OmniToken2WavModel_Export
from modeling_export import Qwen2_5OmniModel_Export, Qwen2_5OmniVisionEncoder_Export
from axengine import InferenceSession
import onnxruntime as ort
from ml_dtypes import bfloat16

class ONNXInfer:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
    def forward(self, inputs):
   
        outputs = self.session.run(None, inputs)
        return outputs

class AXModelInfer:
    def __init__(self, axmodel_path):
        self.session = InferenceSession(axmodel_path, providers=["CPUExecutionProvider"])
        
    def forward(self, inputs, shape_group=None):
        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs

def post_process(data, topk=1, topp=0.001, temperature=0.1):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft


class AXLMModelInfer:
    def __init__(self, mode_dir, model_name, prefill_len, lastN):
       
        # model_name="qwen2_5_omni_text"

        self.prefill_len = prefill_len
        

        self.cfg = AutoConfig.from_pretrained(
            mode_dir, trust_remote_code=True
        )

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size
        
        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            session = InferenceSession(
                f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel"
            )
            self.prefill_decoder_sessins.append(session)
        self.post_process_session = InferenceSession(
            f"{mode_dir}/{model_name}_post.axmodel"
        )

        self.embeds = np.load(f"{mode_dir}/model.embed_tokens.weight.npy")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_dir, trust_remote_code=True
        )
        self.lastN = lastN

        
    
    def forward(self, input_ids, start_token_id, vit_output, position_ids):

        kv_dim = self.cfg.hidden_size // self.cfg.num_attention_heads * self.cfg.num_key_value_heads
        k_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]

        token_ids = input_ids.squeeze().tolist()
        
        prefill_data = np.take(self.embeds, token_ids, axis=0)
        prefill_data = prefill_data.astype(bfloat16)

        image_start_index = np.where(np.array(token_ids) == start_token_id)[0].tolist()[0]
        print(0)
        image_insert_index = image_start_index + 1

        prefill_data[ image_insert_index : image_insert_index + vit_output.shape[1]] = vit_output[0, :, :]
        token_len = len(prefill_data)

        indices = np.zeros((3, self.prefill_len), dtype=np.uint32)

        indices[:, 0:token_len] = position_ids.squeeze(1).astype(np.uint32)
        print("indices", indices.shape)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(bfloat16)
        print("data",data.shape)
        data[:, 0:token_len] = prefill_data
        for i in range(token_len):
            mask[:, i, : i + 1] = 0

        mask = mask.astype(bfloat16)
        for i in range(self.num_hidden_layers):
            input_feed = {
                "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=1)

            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data[:, 0:token_len] = outputs[2][:, :token_len, :]

        post_out = self.post_process_session.run(None, {"input": data[:, token_len - 1:token_len, :]})[0]

        
        next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
        posibles = [self.tokenizer.decode([t]) for t in posssible_tokens]
        posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
        token_ids.append(next_token)
        print("posibles",posibles)
        print(self.tokenizer.decode(token_ids[-1:]))
        print("prefill done!")
    
        # set to decoder
        
        start_ids = np.max(indices) + 1
        mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.lastN] -= 65536
        mask[:, :, :token_len] = 0
        for start_indice in range( np.max(indices) + 1, self.lastN + 1):
            
            if self.prefill_len > 0 and start_indice < token_len:
                continue
            next_token = token_ids[start_indice]
            indices = np.array([start_ids], np.uint32).reshape((1, 1))
            start_ids += 1
            data = self.embeds[next_token, :].reshape((1, 1, self.hidden_size)).astype(bfloat16)

            for i in range(self.cfg.num_hidden_layers):
                # print("decode layer:",i)
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                
                outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
                
                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len - 1:
                pass
            else:
                
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, posssible_tokens, possible_soft = post_process(post_out)
                token_ids.append(next_token)
                print(self.tokenizer.decode(token_ids[-1:]))
            if next_token == self.tokenizer.eos_token_id:
                # print("hit eos!")
                break
        
        return self.tokenizer.decode(token_ids[token_len:])
    

class Qwen2_5OmniThinkerForConditionalGeneration_AXInfer(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)

       
        self.audio_tower = ONNXInfer("audio_tower.onnx")
        self.visual = ONNXInfer("Qwen2.5-Omni-3B_vision.onnx")
        self.text_model = AXLMModelInfer("../Qwen2.5-Omni-7B-AX650N/", "qwen2_5_omni_text", 544, 1023)
        self.processor = Qwen2_5OmniProcessor.from_pretrained("../Qwen2.5-Omni-7B-AX650N/") 

    def forward(self, messages):
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {"type":"text", "text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        #         ],
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "video", "video": video_path, "max_pixels": 308 * 308, "min_pixels": 308 * 308, "fps": 1.0,} ,
        #         ],
        #     },
        # ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True, min_pixels=308*308, max_pixel=308*308)
        inputs = inputs.to("cpu").to(torch.float32)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].view(2,484,3,392).permute(0,2,1,3)


class Qwen2_5OmniModel_AXInfer(Qwen2_5OmniModel_Export):
    def __init__(self, config, max_len_talker_generate_codes=600):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_AXInfer(config.thinker_config)

        self.has_talker = config.enable_audio_output
        if config.enable_audio_output:
            self.enable_talker()

        self.max_len_talker_generate_codes = max_len_talker_generate_codes

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel_Export(self.config.token2wav_config) 
        # self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config) 
        self.token2wav.to(self.config.torch_dtype)
        self.has_talker = True