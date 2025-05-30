
import torch 
from torch import nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import onnxruntime as ort
from qwen_omni_utils import process_mm_info
from transformers import AutoTokenizer, AutoConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig
from transformers.image_utils import PILImageResampling
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder, Qwen2_5OmniThinkerConfig,\
Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionSdpaAttention, apply_rotary_pos_emb_vision, Qwen2_5OmniVisionAttention, Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel,\
    kaiser_sinc_filter1d
from audio_export import Qwen2_5OmniAudioEncoder_Export
from token2wav_export import Qwen2_5OmniToken2WavModel_Export
from preprocess import Qwen2VLImageProcessorExport
# from modeling_export import Qwen2_5OmniModel_Export, Qwen2_5OmniVisionEncoder_Export
from axengine import InferenceSession
import onnxruntime as ort
from ml_dtypes import bfloat16
from itertools import accumulate
import operator
from scipy.special import softmax
import gc
import os 

# class ONNXInfer:
#     def __init__(self, onnx_path):
#         self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
#     def __call__(self, inputs):
   
#         outputs = self.session.run(None, inputs)
#         return outputs

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



class AXModelInferStatic:
    def __init__(self, axmodel_path):
        self.session = InferEngine(axmodel_path)
        
    def __call__(self, inputs, shape_group=None):
        print("shape_group",shape_group)
        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs

class AXModelInferDynamic:
    def __init__(self, axmodel_path):
        self.axmodel_path = axmodel_path
        
    def __call__(self, inputs, shape_group=None):
        print("shape_group",shape_group)
        session = InferEngine(self.axmodel_path)
        if shape_group is None:
            outputs = session.run(None, inputs)
        else:
            outputs = session.run(None, inputs, shape_group=shape_group)

        return outputs

class AXModelInfer:
    def __init__(self, axmodel_path, run_dynamic=False):
        if run_dynamic:
            self.model = AXModelInferDynamic(axmodel_path)
        else:
            self.model = AXModelInferStatic(axmodel_path)
    
    def __call__(self, inputs, shape_group=None):
        try:
            outputs = self.model(inputs, shape_group)
        except Exception as e:
            if hasattr(self.model, "axmodel_path"):
                print(f"axmodel_path:{self.model.axmodel_path}")
            print(e)
            raise e

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

    # def softmax(l: np.ndarray) -> np.ndarray:
    #     l_max = l - l.max()
    #     l_exp = np.exp(l_max)
    #     res = l_exp / np.sum(l_exp)
    #     return res.astype(np.float64)

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

def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: List[int],
    grid_hs: List[int],
    grid_ws: List[int],
):
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids

def get_chunked_index(
     token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
) -> list[tuple[int, int]]:
    """
    Splits token index list into chunks based on token value ranges.

    Given a list of token indices, returns a list of (start, end) index tuples representing
    slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

    For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
    - the first chunk contains token values < 1000,
    - the second chunk contains values >= 1000 and < 2000, and so on.

    Parameters:
        token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                            token index values.
        t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
        remove_index (`int`) An index id to subtract from `token_indices` before chunking

    Returns:
        `List[Tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                            and end (exclusive) indices of a chunk in `token_indices`.
    """

    def _iter():
        i, start_idx = 0, 0  # skip bos token
        current_chunk = 1
        while i < len(token_indices):  # skip eos token
            if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                yield (start_idx, i)
                start_idx = i
                current_chunk += 1
            i += 1
        yield (start_idx, len(token_indices))

    return list(_iter())

def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        use_audio_in_video (`bool`, *optional*):
                If set to `True`, use the audio in video.
        audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = 2
    image_token_id = config.image_token_index
    video_token_id = config.video_token_index
    audio_token_id = config.audio_token_index
    vision_start_token_id = config.vision_start_token_id
    audio_start_token_id = config.audio_start_token_id
    position_id_per_seconds = config.position_id_per_seconds
    seconds_per_chunk = config.seconds_per_chunk

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = (
                image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
            )
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )

                    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                    video_chunk_indexes = get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    audio_chunk_indexes = get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    sub_len = 0
                    for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                        video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                        audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                        if video_chunk_index is not None:
                            sub_len += video_chunk_index[1] - video_chunk_index[0]

                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                            )
                        if audio_chunk_index is not None:
                            sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                            )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


class AXLanguageModelInfer:
    def __init__(self, cfg, mode_dir, model_name, prefill_len, lastN, run_dynamic=False):
       
        # model_name="qwen2_5_omni_text"
        self.cfg = cfg
        self.prefill_len = prefill_len

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size
        
        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            # session = InferenceSession(
            #     f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel"
            # )
            session = AXModelInfer(f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel", run_dynamic)
            self.prefill_decoder_sessins.append(session)
        # self.post_process_session = InferenceSession(
        #     f"{mode_dir}/{model_name}_post.axmodel"
        # )
        self.post_process_session = AXModelInfer(f"{mode_dir}/{model_name}_post.axmodel", run_dynamic)
        self.embeds = np.load(f"{mode_dir}/model.embed_tokens.weight.npy")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_dir, trust_remote_code=True
        )
        self.lastN = lastN

        
    
    def __call__(self,token_ids, prefill_data, position_ids):

        kv_dim = self.cfg.hidden_size // self.cfg.num_attention_heads * self.cfg.num_key_value_heads
        k_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]

        # token_ids = input_ids.squeeze().tolist()
        
        # prefill_data = np.take(self.embeds, token_ids, axis=0)
        # prefill_data = prefill_data.astype(bfloat16)

        # image_start_index = np.where(np.array(token_ids) == start_token_id)[0].tolist()[0]
        # print(0)
        # image_insert_index = image_start_index + 1

        # prefill_data[ image_insert_index : image_insert_index + vit_output.shape[1]] = vit_output[0, :, :]
        print("prefill_data.shape",prefill_data.shape)
        token_len = prefill_data.shape[1]

        indices = np.zeros((3, self.prefill_len), dtype=np.uint32)
        print("token_len",token_len)
        print("position_ids.shape",position_ids.shape)
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
            outputs = self.prefill_decoder_sessins[i]( input_feed, shape_group=1)

            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data[:, 0:token_len] = outputs[2][:, :token_len, :]

        post_norm, post_out = self.post_process_session( {"input": data[:, token_len - 1:token_len, :]})

        print("post_out",post_out, post_out.shape)
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
            print("553, indices",indices.shape)
            for i in range(self.cfg.num_hidden_layers):
                # print("decode layer:",i)
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                
                outputs = self.prefill_decoder_sessins[i]( input_feed, shape_group=0)
                
                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len - 1:
                pass
            else:
                
                post_norm, post_out = self.post_process_session( {"input": data})
                print("post_out",post_out.shape)
                next_token, posssible_tokens, possible_soft = post_process(post_out)
                token_ids.append(next_token)
                print(self.tokenizer.decode(token_ids[-1:]))
            if next_token == self.tokenizer.eos_token_id:
                # print("hit eos!")
                break
        
        return self.tokenizer.decode(token_ids[token_len:])
    

class Qwen2_5OmniAudioEncoder_AXInfer:
    def __init__(self, config, axmodel_path):
        self.config = config
        self.model = AXModelInfer(axmodel_path)
        self.n_window = config.n_window
        self.dtype = torch.bfloat16

    def padded_and_mask_function_maxlen(self, tensor_list, tensor_len, max_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        # max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


    def __call__(self, input_features, feature_lens=None, aftercnn_lens=None,):
        
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        print("chunk_num",chunk_num)
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        print("chunk_lengths",chunk_lengths)
        tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
        # tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        print("tail_chunk_index",tail_chunk_index)
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        print("chunk_lengths",chunk_lengths)
        print("input_features",input_features.shape)
        
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function_maxlen(
            chunk_list, chunk_lengths, max_len=self.n_window * 2, padding_value=0, padding_side="right"
        )
        print("padded_feature",padded_feature.shape)
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        # print("padded_mask_after_cnn.sum()",padded_mask_after_cnn.sum())
        # seq_length = padded_mask_after_cnn.sum().item()
        d0,_,d2 =  padded_feature.shape
        padded_len = torch.tensor(d0*d2, dtype=feature_lens.dtype, device=feature_lens.device)
        seq_len, _ = self._get_feat_extract_output_lengths(padded_len)
        # attention_mask = torch.full(
        #     [1, seq_len, seq_len],
        #     -3.3895313892515355e+38,
        #     device=padded_feature.device,
        #     dtype=padded_feature.dtype,
        # )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        attention_mask = torch.zeros(
            [1, seq_len, seq_len], device=padded_feature.device, dtype=torch.float32
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 1

        # torch.save(attention_mask[:, 0:self.n_window, 0:self.n_window], "attention_mask.pth")
        # torch.save(padded_feature[0:1], "padded_feature.pth")
        # torch.save(padded_mask[0:1], "padded_mask.pth")
        print("padded_mask.dtype",padded_mask.dtype)
        print("padded_feature",padded_feature.dtype)
        print("attention_mask",attention_mask.dtype)
        token_audio_list = []
        for di in range(d0):
            # print("padded_feature[di:di+1]",padded_feature[di:di+1].shape)

            inputs = {"padded_feature": padded_feature[di:di+1].to(torch.float32).cpu().numpy(),
                    "padded_mask": padded_mask[di:di+1].to(torch.int32).cpu().numpy(),
                    "attention_mask":attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window].to(torch.float32).cpu().numpy()}
            token_audio = self.model(inputs)[0]
            token_audio = torch.from_numpy(token_audio)
            # print("token_audio",token_audio.shape)
            token_audio_list.append(token_audio)
        token_audio = torch.cat(token_audio_list, 0)
        # print("token_audio",token_audio.shape)
        _, output_lens = self._get_feat_extract_output_lengths(feature_lens)
        output_lens = output_lens.item()
        token_audio = token_audio[0:output_lens]
        # if not return_dict:
            # return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
        return token_audio


class Qwen2_5OmniVisionEncoder_AXInfer:
    def __init__(self, config, axmodel_path):
        self.config = config 
        self.model = AXModelInfer(axmodel_path)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    # def __call__(self, hidden_states, grid_thw):
    #     t, channel, grid_hw,  tpp  = hidden_states.shape                # 2,3,484,392
    #     # t, grid_hw,  tpp, channel  = hidden_states.shape
    #     # hidden_states = hidden_states.permute(0,1,3,2).reshape(t*grid_hw, channel*tpp)

    #     assert grid_thw.shape[0]==1, f"not support shape:{grid_thw.shape}"

    #     t,grid_h,grid_w = grid_thw[0]

    #     rotary_pos_emb = self.rot_pos_emb(grid_thw)                         # 968,40
    #     window_index, cu_window_seqlens = self.get_window_index(grid_thw)   # window_index shape 242,   cu_window_seqlens: [0, 64, 128, 176, 240, 304, 352, 400, 448, 484, 548, 612, 660, 724, 788, 836, 884, 932, 968]

    #     cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    #         dim=0,
    #         # Select dtype based on the following factors:
    #         #  - FA2 requires that cu_seqlens_q must have dtype int32
    #         #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
    #         # See https://github.com/huggingface/transformers/pull/34852 for more information
    #         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    #     )
    #     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
    #     print("hidden_states.shape",hidden_states.shape)    # grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size

        
    #     llm_grid_h, llm_grid_w = (
    #             grid_h // self.spatial_merge_size,
    #             grid_w // self.spatial_merge_size,
    #         )
    #     vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
    #     pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
    #     pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
    #     num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
    #     num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

    #     # thw, dim = hidden_states.shape
    #     # hidden_states = hidden_states.view(t, -1, dim)
        
    #     torch.save(window_index[0:llm_grid_h*llm_grid_w], "window_index.pth")
        
    #     torch.save(cu_seqlens[0:2], "cu_seqlens.pth")
        

    #     win_idx_t = window_index[0:llm_grid_h*llm_grid_w]

    #     cu_win_seqlens_t = cu_window_seqlens[0 : 1+ num_windows_h*num_windows_w]
    #     cu_seqlens_t = cu_seqlens[0:2]

    #     cu_win_seqlens_t = torch.tensor(
    #                                     cu_win_seqlens_t,
    #                                     device=hidden_states.device,
    #                                     dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    #                                     )
    #     cu_win_seqlens_t = torch.unique_consecutive(cu_win_seqlens_t)
    #     torch.save(cu_win_seqlens_t, "cu_window_seqlens.pth")
        
    #     rope_t = rotary_pos_emb[0: grid_h*grid_w]
    
    #     seq_len = grid_hw
        

    #     rope_t = rope_t.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    #     rope_t = rope_t[win_idx_t, :, :]
    #     rope_t = rope_t.reshape(seq_len, -1)

    #     torch.save(rope_t, "rotary_pos_emb.pth")

    #     emb = torch.cat((rope_t, rope_t), dim=-1)
    #     pos_embs = (emb.cos(), emb.sin())


    #     out = []
    #     for ti in range(t):
    #         ht = hidden_states[ti:ti+1]
    #         inputs = {"hidden_states": ht.cpu().numpy()}
    #         ht = self.model(inputs)[0]
    #         ht = torch.from_numpy(ht)

    #         out.append(ht)
    #     out = torch.cat(out, 0)

        

    #     return out

    def __call__(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        
        t = hidden_states.shape[0]
        print("h shape",hidden_states.shape)
        outputs = []
        for ti in range(t):
            ht = hidden_states[ti:ti+1]

            inputs = {"hidden_states": ht.cpu().to(torch.uint8).numpy(),}
            out = self.model( inputs)[0]
            out = torch.from_numpy(out).to(grid_thw.device)
            outputs.append(out)
        outputs = torch.cat(outputs, 0)
        return outputs

class Qwen2_5OmniThinkerForConditionalGeneration_AXInfer:
    def __init__(self, config: Qwen2_5OmniThinkerConfig, run_dynamic=False):
      
        self.config = config
        self.audio_tower = Qwen2_5OmniAudioEncoder_AXInfer(config.audio_config, "../audio_tower.axmodel")
        self.visual = Qwen2_5OmniVisionEncoder_AXInfer(config.vision_config, "../Qwen2.5-Omni-7B_vision.axmodel")
        self.text_model = AXLanguageModelInfer(config.text_config, "../../Qwen2.5-Omni-3B-AX650N-prefill352/", "qwen2_5_omni_text", 352, 1023, run_dynamic)
        self.processor = Qwen2_5OmniProcessor.from_pretrained("../../Qwen2.5-Omni-3B-AX650N-prefill352/") 

    def __call__(self, messages):
        print("start")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("text",text)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        print("609")
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True, min_pixels=308*308, max_pixel=308*308)
        # inputs.keys dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features'])
        print("inputs",inputs.keys())
        inputs = inputs.to("cpu").to(torch.float32)
        # pixel_values_videos = inputs.get("pixel_values_videos", None)

        # pixel_values_videos = pixel_values_videos.view(2,484,3,392).permute(0,2,1,3) # NCHW for onnx
        # pixel_values_videos = pixel_values_videos.view(2,484,3,392).permute(0,1,3,2)  # 2,484,392,3  NHWC for axmodel
        img_processor = Qwen2VLImageProcessorExport(max_pixels=308*308, patch_size=14, temporal_patch_size=2, merge_size=2)

        image_mean = [
            0.48145466,
            0.4578275,
            0.40821073
        ]

        image_std =  [
            0.26862954,
            0.26130258,
            0.27577711
        ]
        
        pixel_values, grid_thw = img_processor._preprocess(videos[0], do_resize=True, resample=PILImageResampling.BICUBIC, 
                                            do_rescale=False, do_normalize=False, 
                                            do_convert_rgb=True)

        t,seq_len,tpp,_ = pixel_values.shape

        pixel_values = torch.from_numpy(pixel_values)
        # mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
        
        # std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
        
        # pixel_values = (pixel_values-mean)/std

        # pixel_values = pixel_values.permute(0,3,1,2)

        pixel_values_videos = pixel_values

        
        input_ids = inputs["input_ids"]
        inputs_embeds = np.take(self.text_model.embeds, input_ids, axis=0)
        inputs_embeds = torch.from_numpy(inputs_embeds)

        input_features = inputs["input_features"]
        print("input_features",input_features)
        feature_attention_mask = inputs.get("feature_attention_mask", None)
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None


        if input_features is not None:
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            )
            feature_lens = (
                audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            )
            audio_features = self.audio_tower(
                input_features,
                feature_lens=feature_lens,
                aftercnn_lens=audio_feat_lengths,
            )

            audio_mask = (
                    (input_ids == self.config.audio_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        
            print("audio_features",audio_features)
            del audio_features
            del self.audio_tower
            gc.collect()
        
        image_grid_thw = inputs.get("image_grid_thw", None)
        video_grid_thw = inputs.get("video_grid_thw", None)
        attention_mask = inputs.get("attention_mask", None)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, video_grid_thw)
            print("video_embeds",video_embeds)

            video_mask = (
                        (input_ids == self.config.video_token_index)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                        .to(inputs_embeds.device)
                    )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            del video_embeds
            del self.visual
            gc.collect()

        use_audio_in_video = True
        
        second_per_grids = torch.ones(1)
        print("config",self.config)
        position_ids,_ = get_rope_index(self.config, input_ids, image_grid_thw, video_grid_thw, attention_mask, use_audio_in_video, audio_feature_lengths, second_per_grids)
        print("position_ids",position_ids)
        print("input_ids",input_ids)
        output = self.text_model(input_ids[0].tolist(), inputs_embeds, position_ids.cpu().numpy())

        return output

class Qwen2_5OmniModel_AXInfer:
    def __init__(self, config, max_len_talker_generate_codes=600, run_dynamic=False):
        self.config = config
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_AXInfer(config.thinker_config, run_dynamic=run_dynamic)

    #     self.has_talker = config.enable_audio_output
    #     if config.enable_audio_output:
    #         self.enable_talker()

    #     self.max_len_talker_generate_codes = max_len_talker_generate_codes

    # def enable_talker(self):
    #     self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
    #     self.token2wav = Qwen2_5OmniToken2WavModel_Export(self.config.token2wav_config) 
    #     # self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config) 
    #     self.token2wav.to(self.config.torch_dtype)
    #     self.has_talker = True

    def __call__(self, messages):
        return self.thinker(messages)