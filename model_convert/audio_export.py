import torch 
import torch.nn as nn
import torch.nn.functional as F
import operator
import math
import time
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import onnxruntime as ort
from itertools import accumulate
import onnxruntime as ort
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniModel, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder, Qwen2_5OmniThinkerConfig,\
Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionSdpaAttention, apply_rotary_pos_emb_vision, Qwen2_5OmniAudioEncoder, Qwen2_5OmniAudioAttention, Qwen2_5OmniAudioEncoderLayer,Qwen2_5OmniAudioEncoderConfig


class Qwen2_5OmniAudioAttention_Export(Qwen2_5OmniAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        seq_length, _ = hidden_states.size()

        # get query proj
        # query_states = self.q_proj(hidden_states)
        query_states = (hidden_states @ self.q_proj.weight.t()) + self.q_proj.bias

        query_states = query_states.reshape(seq_length, self.num_heads, -1)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).reshape(seq_length, self.num_heads, -1)
            value_states = self.v_proj(current_states).reshape(seq_length, self.num_heads, -1)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)   # PTQ量化 MSE 较大
        print("attn_weights 72",attn_weights.min().item(), attn_weights.max().item(), attn_weights.mean().item(), attn_weights.std().item())
        # attention_mask = torch.full(
        #     [1, seq_length, key_states.shape[1]],
        #     torch.finfo(query_states.dtype).min,
        #     device=query_states.device,
        #     dtype=query_states.dtype,
        # )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        attn_weights = attn_weights + attention_mask
        print("attn_weights 83",attn_weights.min().item(), attn_weights.max().item(), attn_weights.mean().item(), attn_weights.std().item())
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        print("attn_weights 85",attn_weights.min().item(), attn_weights.max().item(), attn_weights.mean().item(), attn_weights.std().item())
        print("attn_weights",attn_weights)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_output = torch.matmul(attn_weights, value_states).transpose(0, 1).reshape(seq_length, self.embed_dim)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Qwen2_5OmniAudioEncoderLayer_Export(Qwen2_5OmniAudioEncoderLayer):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)

        self.self_attn = Qwen2_5OmniAudioAttention_Export(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = (hidden_states @ self.fc1.weight.t()) + (self.fc1.bias if self.fc1.bias is not None else 0)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = (hidden_states @ self.fc2.weight.t()) + (self.fc2.bias if self.fc2.bias is not None else 0)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Qwen2_5OmniAudioEncoder_Export(Qwen2_5OmniAudioEncoder):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer_Export(config) for _ in range(config.encoder_layers)])

    def forward_static(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        print("run forward_static")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        # hidden_states = padded_embed[padded_mask_after_cnn]
        d0,d1,d2 = padded_embed.shape
        hidden_states = padded_embed.reshape(d0*d1, d2)
        print("hidden_states",hidden_states.shape)
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        seq_length = hidden_states.shape[0]
        attention_mask = torch.full(
            [1, seq_length, seq_length],
            -3.3895313892515355e+38,
            device=padded_feature.device,
            dtype=padded_feature.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        tmp_hidden_states = []
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None and head_mask.size()[0] != (len(self.layers)):
            raise ValueError(
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        cu_seqlens,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                tmp_hidden_states.append(hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states_list = hidden_states.split([seq_length], dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = torch.cat(token_audio_list, dim=0)
        
        _, output_lens = self._get_feat_extract_output_lengths(feature_lens)
        output_lens = output_lens.item()

        if output_hidden_states:
            encoder_states = encoder_states + (token_audio,)
            encoder_states = encoder_states[0:output_lens]
        
        token_audio = token_audio[0:output_lens]

        if not return_dict:
            return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=token_audio, hidden_states=encoder_states, attentions=all_attentions)

    def forward_export(self, padded_feature, padded_mask, attention_mask):

        # save calibration data
        dir_calib_audio = "calib_audio"
        os.makedirs(dir_calib_audio, exist_ok=True)
        time_str = str(time.time())
        np.save(f"{dir_calib_audio}/padded_feature_{time_str}.npy", padded_feature.float().cpu().numpy())
        np.save(f"{dir_calib_audio}/padded_mask_{time_str}.npy", padded_mask.float().cpu().numpy())
        np.save(f"{dir_calib_audio}/attention_mask_{time_str}.npy", attention_mask.float().cpu().numpy())

        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        # hidden_states = padded_embed[padded_mask_after_cnn]
        d0,d1,d2 = padded_embed.shape
        hidden_states = padded_embed.reshape(d0*d1, d2)
        # cu_seqlens = torch.cat(
        #     (
        #         torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
        #         padded_mask_after_cnn.sum(1).cumsum(0),
        #     )
        # ).to(torch.int32)

        # seq_length = hidden_states.shape[0]
        # attention_mask = torch.full(
        #     [1, seq_length, seq_length],
        #     -3.3895313892515355e+38,
        #     device=hidden_states.device,
        #     dtype=hidden_states.dtype,
        # )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        # encoder_states = None
        # all_attentions = None

        tmp_hidden_states = []
        # check if head_mask has a correct number of layers specified if desired
        # if head_mask is not None and head_mask.size()[0] != (len(self.layers)):
        #     raise ValueError(
        #         f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        #     )

        for idx, encoder_layer in enumerate(self.layers):
            # if output_hidden_states:
            #     encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                
                # print("hidden_states",hidden_states.shape)
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    # layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    layer_head_mask=None,
                    output_attentions=False,
                )

                hidden_states = layer_outputs[0]
                tmp_hidden_states.append(hidden_states)

            # if output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # hidden_states_list = hidden_states.split([d0*d1], dim=0)
        # token_audio_list = []
        # for each_audio_states in hidden_states_list:
        #     each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
        #     each_audio_states = self.ln_post(each_audio_states)
        #     each_audio_states = self.proj(each_audio_states)
        #     token_audio_list.append(each_audio_states)
        # token_audio = torch.cat(token_audio_list, dim=0)


        # hidden_states = self.avg_pooler(hidden_states.transpose(0, 1)).transpose_(0, 1)
        hidden_states = self.avg_pooler(hidden_states.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose_(0, 1)
        hidden_states = self.ln_post(hidden_states)
        token_audio = self.proj(hidden_states)

        # if output_hidden_states:
        #     encoder_states = encoder_states + (token_audio,)

        return token_audio
        
    

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("return_dict",return_dict)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )

        assert head_mask is None
        assert not output_attentions  
        assert not output_hidden_states
        print("padded_feature.dtype",padded_feature.dtype)
        print("padded_mask",padded_mask.dtype)
        print("padded_mask_after_cnn",padded_mask_after_cnn.dtype)
        print("aftercnn_lens",aftercnn_lens.dtype)
        # padded_feature = padded_feature.to(torch.float32)
        # padded_mask = padded_mask.to(torch.int32)
        # padded_mask_after_cnn = padded_mask_after_cnn.to(torch.bool)
        # aftercnn_lens = aftercnn_lens.to(torch.int32)
        

        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        print("padded_mask_after_cnn.sum()",padded_mask_after_cnn.sum())
        # seq_length = padded_mask_after_cnn.sum().item()
        d0,_,d2 =  padded_feature.shape
        padded_len = torch.tensor(d0*d2, dtype=feature_lens.dtype, device=feature_lens.device)
        seq_len, _ = self._get_feat_extract_output_lengths(padded_len)
        attention_mask = torch.full(
            [1, seq_len, seq_len],
            -3.3895313892515355e+38,
            device=padded_feature.device,
            dtype=padded_feature.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        torch.save(attention_mask[:, 0:self.n_window, 0:self.n_window], "attention_mask.pth")
        torch.save(padded_feature[0:1], "padded_feature.pth")
        torch.save(padded_mask[0:1], "padded_mask.pth")


        token_audio_list = []
        for di in range(d0):
            print("padded_feature[di:di+1]",padded_feature[di:di+1].shape)

            token_audio =  self.forward_export(padded_feature[di:di+1], padded_mask[di:di+1],  attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window])        
            print("token_audio",token_audio.shape)
            token_audio_list.append(token_audio)
        token_audio = torch.cat(token_audio_list, 0)
        print("token_audio",token_audio.shape)
        _, output_lens = self._get_feat_extract_output_lengths(feature_lens)
        output_lens = output_lens.item()
        token_audio = token_audio[0:output_lens]
        # if not return_dict:
            # return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=token_audio, hidden_states=None, attentions=None)

    def padded_and_mask_function_maxlen(self, tensor_list, tensor_len, max_len, padding_value=0, padding_side="right"):
        # max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=tensor_list[0].dtype,
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

    def forward_onnx(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

            feature_lens: [B], torch.LongTensor , mel length

            aftercnn_lens : [B], torch.LongTensor , mel length after cnn

            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("return_dict",return_dict)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        print("chunk_num",chunk_num)
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        print("chunk_lengths",chunk_lengths)
        print("feature_lens",feature_lens)
        tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)
        print("chunk_lengths",chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function_maxlen(
            chunk_list, chunk_lengths, max_len=self.n_window * 2, padding_value=0, padding_side="right"
        )

        assert head_mask is None
        assert not output_attentions  
        assert not output_hidden_states
        print("padded_feature.shape",padded_feature.shape)
        print("padded_mask",padded_mask.shape)
        print("padded_mask_after_cnn",padded_mask_after_cnn.shape)
        print("aftercnn_lens",aftercnn_lens.dtype)
        # padded_feature = padded_feature.to(torch.float32)
        # padded_mask = padded_mask.to(torch.int32)
        # padded_mask_after_cnn = padded_mask_after_cnn.to(torch.bool)
        # aftercnn_lens = aftercnn_lens.to(torch.int32)
        

        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        print("padded_mask_after_cnn.sum()",padded_mask_after_cnn.sum())
        # seq_length = padded_mask_after_cnn.sum().item()
        d0,_,d2 =  padded_feature.shape
        padded_len = torch.tensor(d0*d2, dtype=feature_lens.dtype, device=feature_lens.device)
        seq_len, _ = self._get_feat_extract_output_lengths(padded_len)
        attention_mask = torch.full(
            [1, seq_len, seq_len],
            -3.3895313892515355e+38,
            device=padded_feature.device,
            dtype=padded_feature.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        torch.save(attention_mask[:, 0:self.n_window, 0:self.n_window], "attention_mask.pth")
        torch.save(padded_feature[0:1], "padded_feature.pth")
        torch.save(padded_mask[0:1], "padded_mask.pth")

        print("test audio tower Onnx -------------------")
        session = ort.InferenceSession("audio_tower.onnx", providers=["CPUExecutionProvider"])
        
        
        token_audio_list = []
        for di in range(d0):
            print("padded_feature[di:di+1]",padded_feature[di:di+1].shape)
            print("padded_mask[di:di+1]",padded_mask[di:di+1].shape)
            print("attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window]",attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window].shape)
            # token_audio =  self.forward_export(padded_feature[di:di+1], padded_mask[di:di+1],  attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window])        
            inputs = {"padded_feature": padded_feature[di:di+1].to(torch.float32).cpu().numpy(),
                    "padded_mask": padded_mask[di:di+1].to(torch.int64).cpu().numpy(),
                    "attention_mask":attention_mask[:, di*self.n_window:(di+1)*self.n_window, di*self.n_window:(di+1)*self.n_window].to(torch.float32).cpu().numpy()}
            token_audio = session.run(["token_audio"], inputs)[0]
            token_audio = torch.from_numpy(token_audio).to(padded_feature.device)

            token_audio_list.append(token_audio)
        token_audio = torch.cat(token_audio_list, 0)
        print("token_audio",token_audio.shape)
        _, output_lens = self._get_feat_extract_output_lengths(feature_lens)
        output_lens = output_lens.item()
        token_audio = token_audio[0:output_lens]
        # if not return_dict:
            # return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=token_audio, hidden_states=None, attentions=None)

    # def forward_onnx(
    #     self,
    #     input_features,
    #     feature_lens=None,
    #     aftercnn_lens=None,
    #     head_mask=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    # ):
    #     r"""
    #     Args:
    #         input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
    #             Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
    #             obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
    #             `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
    #             `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
    #             and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]

    #         feature_lens: [B], torch.LongTensor , mel length

    #         aftercnn_lens : [B], torch.LongTensor , mel length after cnn

    #         head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
    #             Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

    #             - 1 indicates the head is **not masked**,
    #             - 0 indicates the head is **masked**.
    #         output_attentions (`bool`, *optional*):
    #             Whether or not to return the attentions tensors of all attention layers. See `attentions` under
    #             returned tensors for more detail.
    #         output_hidden_states (`bool`, *optional*):
    #             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
    #             for more detail.
    #         return_dict (`bool`, *optional*):
    #             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    #     """

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     print("return_dict",return_dict)
    #     chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    #     chunk_lengths = torch.tensor(
    #         [self.n_window * 2] * chunk_num.sum(),
    #         dtype=torch.long,
    #         device=feature_lens.device,
    #     )
    #     tail_chunk_index = list(accumulate(chunk_num.tolist(), func=operator.add, initial=-1))[1:]
    #     chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    #     chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

    #     chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
    #     padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
    #         chunk_list, chunk_lengths, padding_value=0, padding_side="right"
    #     )


    #     cu_seqlens = torch.cat(
    #         (
    #             torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
    #             padded_mask_after_cnn.sum(1).cumsum(0),
    #         )
    #     ).to(torch.int32)

    #     seq_length = padded_mask_after_cnn.sum().item()
    #     attention_mask = torch.full(
    #         [1, seq_length, seq_length],
    #         -3.3895313892515355e+38,
    #         device=padded_feature.device,
    #         dtype=padded_feature.dtype,
    #     )
    #     for i in range(1, len(cu_seqlens)):
    #         attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    #     assert head_mask is None
    #     assert not output_attentions  
    #     assert not output_hidden_states
        
    #     # token_audio =  self.forward_export(padded_feature, padded_mask, padded_mask_after_cnn,aftercnn_lens)        

    #     print("test audio tower Onnx -------------------")
    #     session = ort.InferenceSession("audio_tower.onnx", providers=["CPUExecutionProvider"])
        
    #     inputs = {"padded_feature", padded_feature.cpu().numpy(),
    #                 "padded_mask", padded_mask.cpu().numpy(),
    #                 "padded_mask_after_cnn", padded_mask_after_cnn.cpu().numpy(),
    #                 "aftercnn_lens",aftercnn_lens.cpu().numpy(),
    #                 "attention_mask",attention_mask.cpu().numpy()}

    #     token_audio = session.run(["token_audio"], inputs)[0]
    #     token_audio = torch.from_numpy(token_audio).to(padded_feature.device)

    #     # if not return_dict:
    #         # return tuple(v for v in [token_audio, encoder_states, all_attentions] if v is not None)
    #     return BaseModelOutput(last_hidden_state=token_audio, hidden_states=None, attentions=None)

class Qwen2_5OmniThinkerForConditionalGeneration_Export(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)

        self.audio_tower = Qwen2_5OmniAudioEncoder_Export._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        self.visual = None

class Qwen2_5OmniModel_Export(Qwen2_5OmniModel):
    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_Export(config.thinker_config)
        