import gc
import math
import operator
import os
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16
from preprocess import Qwen2VLImageProcessorExport
from qwen_omni_utils import process_mm_info
from scipy.special import softmax
from torch import nn
from transformers import (AutoConfig, AutoTokenizer, Qwen2_5OmniProcessor,
                          Qwen2_5OmniThinkerConfig)
from transformers.image_utils import PILImageResampling
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniToken2WavModel,
    Qwen2_5OmniVisionAttention, Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder, Qwen2_5OmniVisionSdpaAttention,
    apply_rotary_pos_emb_vision, kaiser_sinc_filter1d)


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
    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(len(t_index), -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(len(t_index), llm_grid_h, -1)
        .flatten()
    )
    t_index = (
        torch.Tensor(t_index)
        .view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .flatten()
        .long()
    )
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
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
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
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
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
            remain_images, remain_videos, remain_audios = (
                image_nums,
                video_nums,
                audio_nums,
            )
            multimodal_nums = (
                image_nums + audio_nums
                if use_audio_in_video
                else image_nums + video_nums + audio_nums
            )
            for _ in range(multimodal_nums):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
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
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = (
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * 1 * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (
                        spatial_merge_size**2
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()
                        * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (
                        spatial_merge_size**2
                    )
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    bos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    audio_llm_pos_ids = (
                        torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t)
                        * second_per_grids[video_idx].cpu().float()
                        * position_id_per_seconds
                    ).long()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )

                    t_ntoken_per_chunk = int(
                        position_id_per_seconds * seconds_per_chunk
                    )
                    video_chunk_indexes = get_chunked_index(
                        video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                    )
                    audio_chunk_indexes = get_chunked_index(
                        audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx
                    )
                    sub_len = 0
                    for j in range(
                        max(len(video_chunk_indexes), len(audio_chunk_indexes))
                    ):
                        video_chunk_index = (
                            video_chunk_indexes[j]
                            if j < len(video_chunk_indexes)
                            else None
                        )
                        audio_chunk_index = (
                            audio_chunk_indexes[j]
                            if j < len(audio_chunk_indexes)
                            else None
                        )
                        if video_chunk_index is not None:
                            sub_len += video_chunk_index[1] - video_chunk_index[0]

                            llm_pos_ids_list.append(
                                video_llm_pos_ids[
                                    :, video_chunk_index[0] : video_chunk_index[1]
                                ]
                            )
                        if audio_chunk_index is not None:
                            sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[
                                    :, audio_chunk_index[0] : audio_chunk_index[1]
                                ]
                            )
                    video_len = video_grid_thw[video_idx].prod() // (
                        spatial_merge_size**2
                    )

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    eos_len = 1
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = (
            position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        )
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[
            0
        ]
        mrope_position_deltas = (
            max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        )

        return position_ids, mrope_position_deltas


class RungeKutta4ODESolver:
    def __init__(self, function, initial_value):
        self.function = function
        self.initial_value = initial_value

        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_step(
        self,
        function,
        time_start,
        time_step,
        time_end,
        value_start,
        function_value_start=None,
    ):
        k1 = (
            function_value_start
            if function_value_start is not None
            else function(time_start, value_start)
        )
        k2 = function(
            time_start + time_step * self._one_third,
            value_start + time_step * k1 * self._one_third,
        )
        k3 = function(
            time_start + time_step * self._two_thirds,
            value_start + time_step * (k2 - k1 * self._one_third),
        )
        k4 = function(time_end, value_start + time_step * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * time_step / 8

    def _compute_step(self, function, time_start, time_step, time_end, value_start):
        function_value_start = function(time_start, value_start)
        return (
            self._rk4_step(
                function,
                time_start,
                time_step,
                time_end,
                value_start,
                function_value_start=function_value_start,
            ),
            function_value_start,
        )

    def _linear_interpolation(
        self, time_start, time_end, value_start, value_end, time_point
    ):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points):
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            delta_value, _ = self._compute_step(
                self.function, time_start, time_step, time_end, current_value
            )
            next_value = current_value + delta_value

            while (
                current_index < len(time_points)
                and time_end >= time_points[current_index]
            ):
                solution[current_index] = self._linear_interpolation(
                    time_start,
                    time_end,
                    current_value,
                    next_value,
                    time_points[current_index],
                )
                current_index += 1

            current_value = next_value

        return solution
