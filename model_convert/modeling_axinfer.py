import gc
import math
import operator
import os
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16
from preprocess import Qwen2VLImageProcessorExport
from qwen_omni_utils import process_mm_info
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
    RungeKutta4ODESolver, apply_rotary_pos_emb_vision, kaiser_sinc_filter1d)
from utils import get_chunked_index, get_llm_pos_ids_for_vision, get_rope_index
from utils_axinfer import AxLMInfer, AxModelInfer




class AxLanguageModelInfer:
    def __init__(
        self, cfg, mode_dir, model_name, prefill_len, lastN, run_dynamic=False
    ):

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
            session = AxModelInfer(
                f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel",
                run_dynamic,
            )
            self.prefill_decoder_sessins.append(session)
        # self.post_process_session = InferenceSession(
        #     f"{mode_dir}/{model_name}_post.axmodel"
        # )
        self.post_process_session = AxModelInfer(
            f"{mode_dir}/{model_name}_post.axmodel", run_dynamic
        )
        self.embeds = np.load(f"{mode_dir}/model.embed_tokens.weight.npy")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_dir, trust_remote_code=True)
        self.lastN = lastN

    def post_process(self, data, topk=1, topp=0.001, temperature=0.1):
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

    def __call__(self, token_ids, prefill_data, position_ids):

        kv_dim = (
            self.cfg.hidden_size
            // self.cfg.num_attention_heads
            * self.cfg.num_key_value_heads
        )
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
        print("prefill_data.shape", prefill_data.shape)
        token_len = prefill_data.shape[1]

        indices = np.zeros((3, self.prefill_len), dtype=np.uint32)
        prompt_ignore_length = token_len

        indices[:, 0:token_len] = position_ids.squeeze(1).astype(np.uint32)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(bfloat16)
        thinker_token_embeds = []
        thinker_hidden_states = []
        
        print("prefill_data",prefill_data)
        data[:, 0:token_len] = prefill_data.numpy().astype(bfloat16)
        thinker_token_embeds.append(prefill_data.numpy())
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
            outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=1)

            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data[:, 0:token_len] = outputs[2][:, :token_len, :]

        # _, post_out = self.post_process_session(
        #     {"input": data[:, token_len - 1 : token_len, :]}
        # )
        print("prefill ht before norm",data[:, 0:token_len])
        post_out = self.post_process_session(
            {"input": data[:, token_len - 1 : token_len, :]}
        )[1]

        post_norm = []
        for ti in range(token_len):

            # pn, _ = self.post_process_session({"input": data[:, ti : ti + 1, :]})
            pn = self.post_process_session({"input": data[:, ti : ti + 1, :]})[0]

            post_norm.append(pn)
           
        post_norm = np.concatenate(post_norm, axis=1)
        print("prefill norm hidden_states",post_norm)
    
        thinker_hidden_states.append(post_norm)
        next_token, posssible_tokens, possible_soft = self.post_process( post_out, topk=1)
        # posibles = [self.tokenizer.decode([t]) for t in posssible_tokens]
        # posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
        token_ids.append(next_token)
        # print("posibles", posibles)
        # print(self.tokenizer.decode(token_ids[-1:]))
        # print("prefill done!")

        # set to decoder

        start_ids = np.max(indices) + 1
        mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.lastN] -= 65536
        mask[:, :, :token_len] = 0
        for start_indice in range(np.max(indices) + 1, self.lastN + 1):

            if self.prefill_len > 0 and start_indice < token_len:
                continue
            next_token = token_ids[start_indice]
            indices = np.array([start_ids], np.uint32).reshape((1, 1))
            start_ids += 1
            data = (
                self.embeds[next_token, :]
                .reshape((1, 1, self.hidden_size))
                .astype(bfloat16)
            )
            thinker_token_embeds.append(data)
            # print("indices", indices)
            for i in range(self.cfg.num_hidden_layers):
                # print("decode layer:",i)
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }

                outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=0)

                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len - 1:
                pass
            else:

                post_norm, post_out = self.post_process_session({"input": data})
                thinker_hidden_states.append(post_norm)
                
                next_token, posssible_tokens, possible_soft = self.post_process( post_out)
                token_ids.append(next_token)
                print(self.tokenizer.decode(token_ids[-1:]))
            if next_token == self.tokenizer.eos_token_id:
                # print("hit eos!")
                break
        
        return (
            self.tokenizer.decode(token_ids[token_len:]),
            token_ids[token_len:],
            thinker_token_embeds,
            thinker_hidden_states,
        )


class Qwen2_5OmniAudioEncoder_AXInfer:
    def __init__(self, config, axmodel_path):
        self.config = config
        self.model = AxModelInfer(axmodel_path)
        self.n_window = config.n_window
        self.dtype = torch.bfloat16

    def padded_and_mask_function_maxlen(
        self, tensor_list, tensor_len, max_len, padding_value=0, padding_side="right"
    ):
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

    def __call__(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):

        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        print("chunk_num", chunk_num)
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        print("chunk_lengths", chunk_lengths)
        tail_chunk_index = list(
            accumulate(chunk_num.tolist(), func=operator.add, initial=-1)
        )[1:]
        # tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        print("tail_chunk_index", tail_chunk_index)
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        print("chunk_lengths", chunk_lengths)
        print("input_features", input_features.shape)

        padded_feature, padded_mask, padded_mask_after_cnn = (
            self.padded_and_mask_function_maxlen(
                chunk_list,
                chunk_lengths,
                max_len=self.n_window * 2,
                padding_value=0,
                padding_side="right",
            )
        )
        print("padded_feature", padded_feature.shape)
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        # print("padded_mask_after_cnn.sum()",padded_mask_after_cnn.sum())
        # seq_length = padded_mask_after_cnn.sum().item()
        d0, _, d2 = padded_feature.shape
        padded_len = torch.tensor(
            d0 * d2, dtype=feature_lens.dtype, device=feature_lens.device
        )
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
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 1

        # torch.save(attention_mask[:, 0:self.n_window, 0:self.n_window], "attention_mask.pth")
        # torch.save(padded_feature[0:1], "padded_feature.pth")
        # torch.save(padded_mask[0:1], "padded_mask.pth")
        print("padded_mask.dtype", padded_mask.dtype)
        print("padded_feature", padded_feature.dtype)
        print("attention_mask", attention_mask.dtype)
        token_audio_list = []
        for di in range(d0):
            # print("padded_feature[di:di+1]",padded_feature[di:di+1].shape)

            inputs = {
                "padded_feature": padded_feature[di : di + 1]
                .to(torch.float32)
                .cpu()
                .numpy(),
                "padded_mask": padded_mask[di : di + 1].to(torch.int32).cpu().numpy(),
                "attention_mask": attention_mask[
                    :,
                    di * self.n_window : (di + 1) * self.n_window,
                    di * self.n_window : (di + 1) * self.n_window,
                ]
                .to(torch.float32)
                .cpu()
                .numpy(),
            }
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
        self.model = AxModelInfer(axmodel_path)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    def __call__(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
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
        print("h shape", hidden_states.shape)
        outputs = []
        for ti in range(t):
            ht = hidden_states[ti : ti + 1]

            inputs = {
                "hidden_states": ht.cpu().to(torch.uint8).numpy(),
            }
            out = self.model(inputs)[0]
            out = torch.from_numpy(out).to(grid_thw.device)
            outputs.append(out)
        outputs = torch.cat(outputs, 0)
        return outputs


class Qwen2_5OmniThinkerForConditionalGeneration_AXInfer:
    def __init__(self, config: Qwen2_5OmniThinkerConfig, run_dynamic=False):

        self.config = config
        self.audio_tower = Qwen2_5OmniAudioEncoder_AXInfer(
            config.audio_config, "../audio_tower.axmodel"
        )
        self.visual = Qwen2_5OmniVisionEncoder_AXInfer(
            config.vision_config, "../Qwen2.5-Omni-7B_vision.axmodel"
        )
        self.text_model = AxLanguageModelInfer(
            config.text_config,
            "../../Qwen2.5-Omni-3B-AX650N-prefill352/",
            "qwen2_5_omni_text",
            352,
            1023,
            run_dynamic,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            "../../Qwen2.5-Omni-3B-AX650N-prefill352/"
        )

    def __call__(self, messages):
        print("start")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("text", text)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        print("609")
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
            min_pixels=308 * 308,
            max_pixel=308 * 308,
        )
        # inputs.keys dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features'])
        print("inputs", inputs.keys())
        inputs = inputs.to("cpu").to(torch.float32)
        # pixel_values_videos = inputs.get("pixel_values_videos", None)

        # pixel_values_videos = pixel_values_videos.view(2,484,3,392).permute(0,2,1,3) # NCHW for onnx
        # pixel_values_videos = pixel_values_videos.view(2,484,3,392).permute(0,1,3,2)  # 2,484,392,3  NHWC for axmodel
        img_processor = Qwen2VLImageProcessorExport(
            max_pixels=308 * 308, patch_size=14, temporal_patch_size=2, merge_size=2
        )

        image_mean = [0.48145466, 0.4578275, 0.40821073]

        image_std = [0.26862954, 0.26130258, 0.27577711]

        pixel_values, grid_thw = img_processor._preprocess(
            videos[0],
            do_resize=True,
            resample=PILImageResampling.BICUBIC,
            do_rescale=False,
            do_normalize=False,
            do_convert_rgb=True,
        )

        t, seq_len, tpp, _ = pixel_values.shape

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
        print("input_features", input_features)
        feature_attention_mask = inputs.get("feature_attention_mask", None)
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        if input_features is not None:
            audio_feat_lengths, audio_output_lengths = (
                self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths
                    if audio_feature_lengths is not None
                    else feature_attention_mask.sum(-1)
                )
            )
            feature_lens = (
                audio_feature_lengths
                if audio_feature_lengths is not None
                else feature_attention_mask.sum(-1)
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
            audio_features = audio_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            print("audio_features", audio_features)
            del audio_features
            del self.audio_tower
            gc.collect()

        image_grid_thw = inputs.get("image_grid_thw", None)
        video_grid_thw = inputs.get("video_grid_thw", None)
        attention_mask = inputs.get("attention_mask", None)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, video_grid_thw)
            print("video_embeds", video_embeds)

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
        # print("config", self.config)
        position_ids, _ = get_rope_index(
            self.config,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_feature_lengths,
            second_per_grids,
        )
        # print("position_ids", position_ids)
        print("thinker text model input_ids", input_ids)
        print("thinker text model inputs_embeds",inputs_embeds)
        output, output_token_ids, thinker_token_embeds, thinker_hidden_states = (
            self.text_model(
                input_ids[0].tolist(), inputs_embeds, position_ids.cpu().numpy()
            )
        )

        return (
            output,
            thinker_token_embeds,
            thinker_hidden_states,
            input_ids,
            output_token_ids,
        )


class Qwen2_5OmniTalkerForConditionalGeneration_AXInfer:
    def __init__(self, cfg, run_dynamic=False):

        model_dir = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/"
        self.thinker_to_talker_proj_336 = AxModelInfer(
            f"thinker_to_talker_proj_336.onnx"
        )
        self.thinker_to_talker_proj = AxModelInfer(
            f"thinker_to_talker_proj.onnx"
        )
        self.model = AxLMInfer(
            cfg, model_dir, "qwen2_5_omni_talker", 352, 1023, run_dynamic
        )
        self.text_eos_token = 151861
        self.text_pad_token = 151859
        self.codec_mask_token = 8296
        self.codec_pad_token = 8292
        self.codec_bos_token = 8293
        self.config = cfg

    def __call__(
        self,
        input_ids,
        input_text_ids,
        thinker_reply_part,
        inputs_embeds,
        attention_mask,
        suppress_tokens,
    ):
        image_grid_thw = None
        video_grid_thw = torch.tensor([[2, 22, 22]])
        use_audio_in_video = True
        audio_feature_lengths = torch.tensor([160])
        second_per_grids = torch.tensor([1])
        position_ids, _ = get_rope_index(
            self.config,
            input_text_ids.long(),
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_feature_lengths,
            second_per_grids,
        )
        print("position_ids", position_ids, position_ids.shape)
        # with open("position_ids_talker_335","rb") as f:
        #     position_ids = dill.load(f)
        # print("position_ids", position_ids, position_ids.shape)
        inputs_embeds[:, -1, :] += torch.from_numpy(
            self.model.embeds[self.codec_bos_token]
        )
        inputs_embeds[:, -2, :] += torch.from_numpy(
            self.model.embeds[self.codec_pad_token]
        )
        print(f"model.embeds[{self.codec_bos_token}]", torch.from_numpy(
            self.model.embeds[self.codec_bos_token]
        ))
        print(f"model.embeds[{self.codec_pad_token}]", torch.from_numpy(
            self.model.embeds[self.codec_pad_token]
        ))

        print("inputs_embeds", inputs_embeds)
        
        if inputs_embeds.shape[1] == 336:
            talker_lm_input = self.thinker_to_talker_proj_336(
                {"input": inputs_embeds.numpy()}
            )[0]
        elif inputs_embeds.shape[1] == 1:
            talker_lm_input = self.thinker_to_talker_proj(
                {"input": inputs_embeds.numpy()}
            )[0]
        else:
            assert False

        outputs = self.model(input_embeds=talker_lm_input, position_ids=position_ids, thinker_reply_part=thinker_reply_part)
        return outputs


class Qwen2_5OmniToken2WavDiTModel_AxInfer:
    def __init__(self, config, run_dynamic=False):
        self.model = AxModelInfer("token2wav_dit.onnx", run_dynamic)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats

    def forward_onnx(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
    ):

        inputs = {
            "x": x.cpu().numpy(),
            "cond": cond.cpu().numpy(),
            "spk": spk.cpu().numpy(),
            "code": code.cpu().numpy(),
            "time": time.repeat(x.shape[0]).cpu().numpy(),
        }
        out = self.model(inputs)[0]
        out = torch.from_numpy(out).to(x.device)
        return out

    @torch.no_grad()
    def sample(
        self,
        cond,
        ref_mel,
        code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        y_all = torch.randn([1, 30000, self.mel_dim], dtype=ref_mel.dtype)
        max_duration = code.shape[1] * self.repeats
        y0 = y_all[:, :max_duration].to(code.device)
        batch = ref_mel.shape[0]
        cond = cond.unsqueeze(1).repeat(1, max_duration, 1)
        assert batch == 1, "only support batch size = 1 currently"

        def fn(t, x):
            if guidance_scale < 1e-5:
                pred = self.forward_onnx(x=x, spk=cond, cond=ref_mel, code=code, time=t)
                return pred

            out_put = self.forward_onnx(
                x=x,
                code=code,
                spk=cond,
                cond=ref_mel,
                time=t,
            )
            pred, null_pred = torch.chunk(out_put, 2, dim=0)

            return pred + (pred - null_pred) * guidance_scale

        t_start = 0
        t = torch.linspace(t_start, 1, num_steps, device=code.device, dtype=cond.dtype)
        if sway_coefficient is not None:
            t = t + sway_coefficient * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = RungeKutta4ODESolver(function=fn, initial_value=y0)
        trajectory = solver.integrate(t)

        generated = trajectory[-1]
        generated_mel_spec = generated.permute(0, 2, 1)
        return generated_mel_spec


class Qwen2_5OmniToken2WavBigVGANModel_AxInfer:
    def __init__(self, config, run_dynamic=False):
        self.model = AxModelInfer("token2wav_bigvgan.onnx", run_dynamic)

    def __call__(self, apm_mel):
        import time

        t1 = time.time()
        inputs = {"apm_mel": apm_mel.cpu().numpy()}
        out = self.model(inputs)[0]
        t2 = time.time()
        print("Qwen2_5OmniToken2WavBigVGANModel Onnx infer time", t2 - t1)
        out = torch.from_numpy(out).to(apm_mel.device)
        return out


class Qwen2_5OmniToken2WavModel_AxInfer:
    def __init__(self, config, run_dynamic=False):
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel_AxInfer(
            config.dit_config, run_dynamic
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel_AxInfer(
            config.bigvgan_config, run_dynamic
        )

    def __call__(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        """Generates a waveform from input code and conditioning parameters."""

        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = self.code2wav_bigvgan_model(mel_spectrogram)
        return waveform


class Qwen2_5OmniModel_AXInfer:
    def __init__(
        self,
        config,
        max_len_talker_generate_codes=600,
        run_dynamic=False,
        enable_audio_output=True,
    ):
        self.config = config
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_AXInfer(
            config.thinker_config, run_dynamic=False
        )

        self.has_talker = enable_audio_output
        self.speaker_map = {}
        if enable_audio_output:
            self.enable_talker(run_dynamic)
            self.load_speakers("../../Qwen2.5-Omni-3B/spk_dict.pt")

        self.max_len_talker_generate_codes = max_len_talker_generate_codes

    def enable_talker(self, run_dynamic):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration_AXInfer(
            self.config.talker_config, False
        )
        self.token2wav = Qwen2_5OmniToken2WavModel_AxInfer(
            self.config.token2wav_config, run_dynamic
        )
        self.has_talker = True

    def load_speakers(self, path):
        for key, value in torch.load(path).items():
            self.speaker_map[key] = value
        print("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def __call__(
        self,
        messages,
        speaker: str = "Chelsie",
    ):

        # if not os.path.exists("thinker_result"):
        (
            thinker_result,
            thinker_token_embeds,
            thinker_hidden_states,
            input_ids,
            output_token_ids,
        ) = self.thinker(messages)

        speaker_params = self.speaker_map[speaker]
        print(thinker_result)
        print("len thinker_token_embeds", len(thinker_token_embeds))
        print("len thinker_token_embeds[0]", len(thinker_token_embeds[0]))
        print("thinker_token_embeds[0][0].shape", thinker_token_embeds[0].shape)
        print("thinker_token_embeds[1][0].shape", thinker_token_embeds[1].shape)

        print("len thinker_hidden_states", len(thinker_hidden_states))
        print("len thinker_hidden_states[0]", len(thinker_hidden_states[0]))
        print("thinker_hidden_states[0][0].shape", thinker_hidden_states[0].shape)
        print("thinker_hidden_states[1][0].shape", thinker_hidden_states[1].shape)

        # thinker_token_embeds = torch.from_numpy(thinker_token_embeds)
        # thinker_hidden_states = torch.from_numpy(thinker_hidden_states)
        thinker_hidden_states = [
            torch.from_numpy(ht.astype(np.float32)) for ht in thinker_hidden_states
        ]
        thinker_token_embeds = [
            torch.from_numpy(ht.astype(np.float32)) for ht in thinker_token_embeds
        ]

        print("thinker_token_embeds[0]  embeds_to_talker ",thinker_token_embeds[0])
        print("thinker_hidden_states[0]",thinker_hidden_states[0])
        
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(
            thinker_token_embeds[1:], dim=1
        )
        print("thinker_reply_part",thinker_reply_part.shape)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = speaker_params["bos_token"]
        talker_text_bos_embed = self.thinker.text_model.embeds[talker_text_bos_token]
        talker_text_bos_embed = torch.from_numpy(talker_text_bos_embed).reshape(
            1, 1, -1
        )

        print("talker_inputs_embeds", talker_inputs_embeds.shape)
        print("talker_text_bos_embed", talker_text_bos_embed.shape)
        print("thinker_reply_part[:, :1, :]", thinker_reply_part[:, :1, :].shape)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = self.thinker.text_model.embeds[self.talker.text_eos_token]
        eos_embedding = torch.from_numpy(eos_embedding).reshape(1, 1, -1)

        pad_embedding = self.thinker.text_model.embeds[self.talker.text_pad_token]
        pad_embedding = torch.from_numpy(pad_embedding).reshape(1, 1, -1)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )
        print("thinker_reply_part",thinker_reply_part.shape)
        # input_ids = torch.from_numpy(input_ids)
        print("input_ids", input_ids.shape)

        thinker_generate_ids = torch.tensor(output_token_ids).reshape(1, -1)
        print("thinker_generate_ids",thinker_generate_ids)
        print("thinker_generate_ids.shape", thinker_generate_ids.shape)
        talker_input_text_ids = torch.cat(
            [
                input_ids,
                torch.tensor(
                    [[talker_text_bos_token]],
                    dtype=torch.long,
                ),
                thinker_generate_ids[:,:1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(
                    input_ids,
                    fill_value=self.talker.codec_mask_token,
                ),
                torch.tensor(
                    [[self.talker.codec_pad_token]],
                    dtype=torch.long,
                ),
                torch.tensor(
                    [[self.talker.codec_bos_token]],
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

        talker_attention_mask = torch.cat(
            [torch.ones(1, 334), torch.ones(1, 334).new_ones((1, 2))], dim=1
        )

        with open("talker_input_ids", "rb") as f:
                talker_input_ids_gt = dill.load(f).float()

        with open("talker_input_text_ids", "rb") as f:
                talker_input_text_ids_gt = dill.load(f).float()

        with open("thinker_reply_part", "rb") as f:
                thinker_reply_part_gt =dill.load(f).float()

        with open("talker_inputs_embeds", "rb") as f:
                talker_inputs_embeds_gt = dill.load(f).float()

        with open("talker_attention_mask", "rb") as f:
                talker_attention_mask_gt = dill.load(f).float()
                talker_attention_mask_gt = talker_attention_mask_gt.to(torch.long)


        if not torch.equal(talker_input_ids,talker_input_ids_gt.long()):
            print("talker_input_ids",talker_input_ids)
            print("talker_input_ids_gt",talker_input_ids_gt)

        if not torch.equal(talker_input_text_ids,talker_input_text_ids_gt.long()):
            print("talker_input_text_ids",talker_input_text_ids)
            print("talker_input_text_ids_gt",talker_input_text_ids_gt)

        if not torch.equal(thinker_reply_part,thinker_reply_part_gt):
            print("thinker_reply_part",thinker_reply_part)
            print("thinker_reply_part_gt",thinker_reply_part_gt)

            # thinker_reply_part = thinker_reply_part_gt

        if not torch.equal(talker_inputs_embeds,talker_inputs_embeds_gt):
            print("talker_inputs_embeds",talker_inputs_embeds)
            print("talker_inputs_embeds_gt",talker_inputs_embeds_gt)

        if not torch.equal(talker_attention_mask,talker_attention_mask_gt):
            print("talker_attention_mask",talker_attention_mask.shape)
            print("talker_attention_mask_gt", talker_attention_mask_gt.shape)

        # if os.path.exists("talker_generate_codes"):
        #     with open("talker_generate_codes", "rb") as f:
        #         talker_generate_codes = dill.load(f)
        # else:

        talker_result = self.talker(
            talker_input_ids.long(),
            talker_input_text_ids,
            thinker_reply_part,
            talker_inputs_embeds,
            talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
        )
        print("talker_result",talker_result)
        talker_generate_codes = talker_result[talker_input_ids.shape[1] : -1]
        print("talker_generate_codes",talker_generate_codes)
        talker_generate_codes = torch.tensor(talker_generate_codes).reshape(1, -1)
            # with open("talker_generate_codes", "wb") as f:
            #     dill.dump(talker_generate_codes, f)
        
        del self.thinker 
        del self.talker
        gc.collect()


        effictive_len = talker_generate_codes.shape[1]
        effictive_len = min(effictive_len, self.max_len_talker_generate_codes)
        padded_talker_generate_codes = torch.zeros(
            (1, self.max_len_talker_generate_codes)
        )
        padded_talker_generate_codes[:, 0:effictive_len] = talker_generate_codes[
            :, 0:effictive_len
        ]

        wav = self.token2wav(
            padded_talker_generate_codes.long(),
            conditioning=speaker_params["cond"].float(),
            reference_mel=speaker_params["ref_mel"].float(),
        )
        wav = wav[0 : effictive_len * 480]
        print("wav", wav.shape)
        # return thinker_result, wav.float()
        return "", wav.float()
