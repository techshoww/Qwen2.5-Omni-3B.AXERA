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
from utils_axinfer import AxModelInfer
from utils_lm import Qwen2_5OmniTalkerModel_AXInfer, Qwen2_5OmniThinkerTextModel_AXInfer

class Qwen2_5OmniAudioEncoder_AXInfer:
    def __init__(self, config, axmodel_path, run_dynamic=False):
        self.config = config
        self.model = AxModelInfer(axmodel_path, run_dynamic)
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
        
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        
        tail_chunk_index = list(
            accumulate(chunk_num.tolist(), func=operator.add, initial=-1)
        )[1:]
        # tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)

        padded_feature, padded_mask, padded_mask_after_cnn = (
            self.padded_and_mask_function_maxlen(
                chunk_list,
                chunk_lengths,
                max_len=self.n_window * 2,
                padding_value=0,
                padding_side="right",
            )
        )

        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        d0, _, d2 = padded_feature.shape
        padded_len = torch.tensor(
            d0 * d2, dtype=feature_lens.dtype, device=feature_lens.device
        )
        seq_len, _ = self._get_feat_extract_output_lengths(padded_len)
        
        attention_mask = torch.zeros(
            [1, seq_len, seq_len], device=padded_feature.device, dtype=torch.float32
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 1

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
            token_audio_list.append(token_audio)
        token_audio = torch.cat(token_audio_list, 0)
        
        _, output_lens = self._get_feat_extract_output_lengths(feature_lens)
        output_lens = output_lens.item()
        token_audio = token_audio[0:output_lens]
        
        return token_audio


class Qwen2_5OmniVisionEncoder_AXInfer:
    def __init__(self, config, axmodel_path, run_dynamic=False):
        self.config = config
        self.model = AxModelInfer(axmodel_path, run_dynamic)
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
    def __init__(self, config: Qwen2_5OmniThinkerConfig, model_dir, prefill_len, lastN=1023, run_dynamic=False, lazy_load=False):

        self.config = config
        self.audio_tower = Qwen2_5OmniAudioEncoder_AXInfer(
            config.audio_config, f"{model_dir}/audio_tower.axmodel", run_dynamic=True
        )
        self.visual = Qwen2_5OmniVisionEncoder_AXInfer(
            config.vision_config, f"{model_dir}/Qwen2.5-Omni-7B_vision.axmodel", run_dynamic=True
        )
        self.text_model = Qwen2_5OmniThinkerTextModel_AXInfer(
            config.text_config,
            model_dir,
            "qwen2_5_omni_text",
            prefill_len,
            lastN,
            run_dynamic,
            lazy_load
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_dir
        )

    def __call__(self, messages):
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
       
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
        
        inputs = inputs.to("cpu").to(torch.float32)
        
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
        pixel_values_videos = pixel_values

        input_ids = inputs["input_ids"]
        inputs_embeds = np.take(self.text_model.embeds, input_ids, axis=0)
        inputs_embeds = torch.from_numpy(inputs_embeds)

        input_features = inputs["input_features"]
        
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

            del audio_features
            del self.audio_tower
            gc.collect()

        image_grid_thw = inputs.get("image_grid_thw", None)
        video_grid_thw = inputs.get("video_grid_thw", None)
        attention_mask = inputs.get("attention_mask", None)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, video_grid_thw)

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
    def __init__(self, cfg, model_dir, prefill_len, lastN=1023, run_dynamic=False, lazy_load=False,):

        self.thinker_to_talker_proj_prefill = AxModelInfer(
            f"{model_dir}/thinker_to_talker_proj_prefill_{prefill_len}.onnx"
        )
        self.thinker_to_talker_proj = AxModelInfer(
            f"{model_dir}/thinker_to_talker_proj_decode.onnx"
        )
        self.model = Qwen2_5OmniTalkerModel_AXInfer(
            cfg, model_dir, "qwen2_5_omni_talker", prefill_len, lastN, run_dynamic, lazy_load
        )
        self.prefill_len = prefill_len
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
        
        inputs_embeds[:, -1, :] += torch.from_numpy(
            self.model.embeds[self.codec_bos_token]
        )
        inputs_embeds[:, -2, :] += torch.from_numpy(
            self.model.embeds[self.codec_pad_token]
        )
        
        inputs_embeds = inputs_embeds.numpy()
        size = inputs_embeds.shape[1]
        if size > 1:
            assert size <= self.prefill_len
            x = np.zeros([inputs_embeds.shape[0], self.prefill_len, inputs_embeds.shape[2] ], dtype=inputs_embeds.dtype)
            x[:,0:size] = inputs_embeds
            talker_lm_input = self.thinker_to_talker_proj_prefill(
                {"input": x}
            )[0][:,0:size]
        elif size == 1:
            talker_lm_input = self.thinker_to_talker_proj(
                {"input": inputs_embeds}
            )[0]
        else:
            assert False

        outputs = self.model(input_embeds=talker_lm_input, position_ids=position_ids, thinker_reply_part=thinker_reply_part)
        return outputs


class Qwen2_5OmniToken2WavDiTModel_AxInfer:
    def __init__(self, config, model_path, run_dynamic=False):
        self.model = AxModelInfer(model_path, run_dynamic)
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
    def __init__(self, config, model_path, run_dynamic=False):
        self.model = AxModelInfer(model_path, run_dynamic)

    def __call__(self, apm_mel):
        inputs = {"apm_mel": apm_mel.cpu().numpy()}
        out = self.model(inputs)[0]
        out = torch.from_numpy(out).to(apm_mel.device)
        return out


class Qwen2_5OmniToken2WavModel_AxInfer:
    def __init__(self, config, model_dir, run_dynamic=False):
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel_AxInfer(
            config.dit_config, f"{model_dir}/token2wav_dit.onnx", run_dynamic
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel_AxInfer(
            config.bigvgan_config, f"{model_dir}/token2wav_bigvgan.onnx", run_dynamic=True
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
        thinker_dir, 
        talker_dir,
        prefill_len,
        lastN=1023,
        max_len_talker_generate_codes=600,
        run_dynamic=False,
        lazy_load=False,
        enable_audio_output=True,
    ):
        self.config = config
        self.prefill_len = prefill_len
        self.lastN = lastN
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_AXInfer(
            config.thinker_config, thinker_dir, prefill_len, lastN, run_dynamic, lazy_load
        )

        self.has_talker = enable_audio_output
        self.speaker_map = {}
        if enable_audio_output:
            self.enable_talker(talker_dir, run_dynamic, lazy_load)
            self.load_speakers(f"{talker_dir}/spk_dict.pt")

        self.max_len_talker_generate_codes = max_len_talker_generate_codes

    def enable_talker(self, talker_dir, run_dynamic, lazy_load):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration_AXInfer(
            self.config.talker_config, talker_dir, self.prefill_len, self.lastN, run_dynamic, lazy_load
        )
        self.token2wav = Qwen2_5OmniToken2WavModel_AxInfer(
            self.config.token2wav_config, talker_dir, run_dynamic=run_dynamic
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
        
        thinker_hidden_states = [
            torch.from_numpy(ht.astype(np.float32)) for ht in thinker_hidden_states
        ]
        thinker_token_embeds = [
            torch.from_numpy(ht.astype(np.float32)) for ht in thinker_token_embeds
        ]

        
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(
            thinker_token_embeds[1:], dim=1
        )

        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = speaker_params["bos_token"]
        talker_text_bos_embed = self.thinker.text_model.embeds[talker_text_bos_token]
        talker_text_bos_embed = torch.from_numpy(talker_text_bos_embed).reshape(
            1, 1, -1
        )

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

        thinker_generate_ids = torch.tensor(output_token_ids).reshape(1, -1)
        
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

        talker_result = self.talker(
            talker_input_ids.long(),
            talker_input_text_ids,
            thinker_reply_part,
            talker_inputs_embeds,
            talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
        )
        talker_generate_codes = talker_result[talker_input_ids.shape[1] : -1]
        talker_generate_codes = torch.tensor(talker_generate_codes).reshape(1, -1)
        
        # del self.thinker 
        # del self.talker
        # gc.collect()

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
        
        return thinker_result, wav.float()
        
