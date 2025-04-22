import torch 
from torch import nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import onnxruntime as ort
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniModel, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder, Qwen2_5OmniThinkerConfig,\
Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionSdpaAttention, apply_rotary_pos_emb_vision, Qwen2_5OmniVisionAttention, Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel,\
    kaiser_sinc_filter1d
from audio_export import Qwen2_5OmniAudioEncoder_Export
from token2wav_export import Qwen2_5OmniToken2WavModel_Export

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(1, L, S, dtype=query.dtype).to(attn_mask.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ value

class Qwen2_5OmniVisionAttention_Export(Qwen2_5OmniVisionAttention):
    # def __init__(self, dim: int, num_heads: int = 16) -> None:
    #     super().__init__()
    #     self.num_heads = num_heads
    #     self.head_dim = dim // num_heads
    #     self.q = nn.Linear(dim, dim, bias=True)
    #     self.k = nn.Linear(dim, dim, bias=True)
    #     self.v = nn.Linear(dim, dim, bias=True)
    #     self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # print("Qwen2_5OmniVisionAttention, torch.finfo(q.dtype).min", torch.finfo(q.dtype).min, "q.dtype",q.dtype)
        # attention_mask = torch.full(
        #     [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        # )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5OmniVisionSdpaAttention_Export(Qwen2_5OmniVisionSdpaAttention):
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5OmniVisionBlock_Export(Qwen2_5OmniVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__(config, attn_implementation)
        self.attn = Qwen2_5OmniVisionAttention_Export(
            config.hidden_size, num_heads=config.num_heads
        )

    def forward(self, hidden_states, attention_mask, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
    
class Qwen2_5OmniVisionEncoder_Infer(Qwen2_5OmniVisionEncoder):

    def forward_nchw(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        torch.save(hidden_states, "hidden_states.pth")
        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        torch.save(rotary_pos_emb, "rotary_pos_emb.pth")
        torch.save(cu_seqlens, "cu_seqlens.pth")
        torch.save(cu_window_seqlens, "cu_window_seqlens.pth")
        torch.save(window_index, "window_index.pth")

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def forward_by_second_nchw(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # hidden_states = hidden_states.permute(0,2,3,1)
        t, channel, grid_hw,  tpp  = hidden_states.shape
        # t, grid_hw,  tpp, channel  = hidden_states.shape
        # hidden_states = hidden_states.permute(0,1,3,2).reshape(t*grid_hw, channel*tpp)

        assert grid_thw.shape[0]==1, f"not support shape:{grid_thw.shape}"

        t,grid_h,grid_w = grid_thw[0]

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        print("hidden_states.shape",hidden_states.shape)    # grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size

        
        llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        # thw, dim = hidden_states.shape
        # hidden_states = hidden_states.view(t, -1, dim)
        
        torch.save(window_index[0:llm_grid_h*llm_grid_w], "window_index.pth")
        
        torch.save(cu_seqlens[0:2], "cu_seqlens.pth")
        

        win_idx_t = window_index[0:llm_grid_h*llm_grid_w]

        cu_win_seqlens_t = cu_window_seqlens[0 : 1+ num_windows_h*num_windows_w]
        cu_seqlens_t = cu_seqlens[0:2]

        cu_win_seqlens_t = torch.tensor(
                                        cu_win_seqlens_t,
                                        device=hidden_states.device,
                                        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
                                        )
        cu_win_seqlens_t = torch.unique_consecutive(cu_win_seqlens_t)
        torch.save(cu_win_seqlens_t, "cu_window_seqlens.pth")
        
        rope_t = rotary_pos_emb[0: grid_h*grid_w]
    
        seq_len = grid_hw
        

        rope_t = rope_t.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rope_t = rope_t[win_idx_t, :, :]
        rope_t = rope_t.reshape(seq_len, -1)

        torch.save(rope_t, "rotary_pos_emb.pth")

        emb = torch.cat((rope_t, rope_t), dim=-1)
        pos_embs = (emb.cos(), emb.sin())


        out = []
        for ti in range(t):
            ht = hidden_states[ti:ti+1]
            print("ht.shape",ht.shape)
            torch.save(ht, "hidden_states.pth")
            ht = ht.permute(0,2,3,1) 
            ht = ht.permute(0,1,3,2).reshape(grid_hw, channel*tpp)
            ht = self.patch_embed(ht)
            ht = ht.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            ht = ht[win_idx_t, :, :]
            ht = ht.reshape(seq_len, -1)
            
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens_t
                else:
                    cu_seqlens_now = cu_win_seqlens_t
                
                ht = blk(ht, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rope_t)

            ht = self.merger(ht)
            reverse_indices = torch.argsort(win_idx_t)
            ht = ht[reverse_indices, :]

            out.append(ht)
        out = torch.cat(out, 0)
        # np.save("vit_out.npy", out.cpu().numpy())
        return out
    
class Qwen2_5OmniThinkerForConditionalGeneration_Infer(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)

        self.visual = Qwen2_5OmniVisionEncoder_Infer._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )


class Qwen2_5OmniModel_Infer(Qwen2_5OmniModel):
    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_Infer(config.thinker_config)

def generate_attnmask(seq_length, cu_seqlens):
    # attention_mask = torch.zeros([1, seq_length, seq_length],  dtype=torch.bool)
    # for i in range(1, cu_seqlens.shape[0]):
    #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

    attention_mask = torch.full(
            [1, seq_length, seq_length], -3.3895313892515355e+38, dtype=torch.float32
        )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0


    return attention_mask

class Qwen2_5OmniVisionEncoder_Export(Qwen2_5OmniVisionEncoder):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        h = torch.load("hidden_states.pth","cpu", weights_only=True)
        cu_seqlens = torch.load("cu_seqlens.pth","cpu", weights_only=True)
        cu_window_seqlens = torch.load("cu_window_seqlens.pth","cpu", weights_only=True)

        seq_length = h.shape[0] if h.shape[0]!=1 else h.shape[2]
        # seq_length = h.shape[0] if h.shape[0]!=1 else h.shape[1]
        self.attention_mask = generate_attnmask(seq_length, cu_seqlens)
        self.attention_mask_window = generate_attnmask(seq_length, cu_window_seqlens)

        self.rotary_pos_emb_ = torch.load("rotary_pos_emb.pth","cpu", weights_only=True)

        self.window_index = torch.load("window_index.pth","cpu", weights_only=True)
        self.reverse_indices = torch.argsort(self.window_index)

        self.blocks = nn.ModuleList(
            [Qwen2_5OmniVisionBlock_Export(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def forward_export(self, hidden_states):

        device = hidden_states.device
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)
        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)
        self.reverse_indices = self.reverse_indices.to(device)

        t, channel, seq_len, tpp = hidden_states.shape
        assert t==1 
        hidden_states = hidden_states.permute(0,2,1,3).reshape(t,seq_len, channel*tpp)
        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )

        hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[self.reverse_indices, :]

        return hidden_states

    def forward_export_by_second_nchw(self, hidden_states):
        hidden_states = hidden_states.permute(0,2,3,1)
        t, grid_hw,  tpp, channel = hidden_states.shape
        print("hidden_states.shape",hidden_states.shape)
        device = hidden_states.device

        hidden_states = hidden_states.permute(0,1,3,2).reshape(grid_hw, channel*tpp)
        
        self.attention_mask = self.attention_mask.to(device)
        self.attention_mask_window = self.attention_mask_window.to(device)

        self.rotary_pos_emb_ = self.rotary_pos_emb_.to(device)
        self.reverse_indices = self.reverse_indices.to(device)

        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[self.window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask_now = self.attention_mask
            else:
                attention_mask_now = self.attention_mask_window

            hidden_states = blk(
                hidden_states,
                attention_mask=attention_mask_now,
                rotary_pos_emb=self.rotary_pos_emb_,
            )

        hidden_states = self.merger(hidden_states)
        # reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[self.reverse_indices, :]

        return hidden_states

    def forward_onnx_nchw(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        
        print("test Vision Encoder Onnx -------------------")
        session = ort.InferenceSession("Qwen2.5-Omni-7B_vision.onnx", providers=["CPUExecutionProvider"])
        
        inputs = {"hidden_states": hidden_states.cpu().to(torch.float32).numpy().astype(np.float32),}
        out = session.run(["hidden_states_out"], inputs)[0]
        out = torch.from_numpy(out).to(grid_thw.device)
      
        return out

    def forward_onnx_by_second_nchw(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        
        print("test Vision Encoder Onnx -------------------")
        session = ort.InferenceSession("Qwen2.5-Omni-7B_vision.onnx", providers=["CPUExecutionProvider"])
        
        
        t = hidden_states.shape[0]
        print("h shape",hidden_states.shape)
        outputs = []
        for ti in range(t):
            ht = hidden_states[ti:ti+1]

            inputs = {"hidden_states": ht.cpu().to(torch.float32).numpy().astype(np.float32),}
            out = session.run(["hidden_states_out"], inputs)[0]
            out = torch.from_numpy(out).to(grid_thw.device)
            outputs.append(out)
        outputs = torch.cat(outputs, 0)
        return outputs
    
class Qwen2_5OmniThinkerForConditionalGeneration_Export(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)

        self.audio_tower = Qwen2_5OmniAudioEncoder_Export._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        self.visual = Qwen2_5OmniVisionEncoder_Export._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )


class Qwen2_5OmniModel_Export(Qwen2_5OmniModel):
    def __init__(self, config, max_len_talker_generate_codes=1000):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_Export(config.thinker_config)

        self.has_talker = config.enable_audio_output
        if config.enable_audio_output:
            self.enable_talker()

        self.max_len_talker_generate_codes = max_len_talker_generate_codes

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel_Export(self.config.token2wav_config) 
        self.token2wav.float()
        self.has_talker = True
        

    def init_upsampler_downsampler(self):
        
        len_resblocks = len(self.token2wav.code2wav_bigvgan_model.resblocks)

        filter = kaiser_sinc_filter1d(0.25, 0.3, 12)
        for i in range(len_resblocks):
            len_act = len(self.token2wav.code2wav_bigvgan_model.resblocks[i].activations)
            for j in range(len_act):
                d0 = self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].downsample.conv.weight.data.shape[0]
                # 冻结卷积层的权重，并加载预计算的滤波器
                self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].downsample.conv.weight.data.copy_(filter.expand(d0, -1, -1))
                self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].downsample.conv.weight.requires_grad = False  # 冻结权重，使其不可训练

                d0 = self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].upsample.conv_transpose.weight.data.shape[0]
                # 冻结卷积层的权重，并加载预计算的滤波器
                self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].upsample.conv_transpose.weight.data.copy_(filter.expand(d0, -1, -1))
                self.token2wav.code2wav_bigvgan_model.resblocks[i].activations[j].upsample.conv_transpose.weight.requires_grad = False  # 冻结权重，使其不可训练


        'token2wav.code2wav_bigvgan_model.activation_post.downsample.conv.weight', 'token2wav.code2wav_bigvgan_model.activation_post.upsample.conv_transpose.weight',

        d0 = self.token2wav.code2wav_bigvgan_model.activation_post.downsample.conv.weight.data.shape[0]
        self.token2wav.code2wav_bigvgan_model.activation_post.downsample.conv.weight.data.copy_(filter.expand(d0, -1, -1))
        self.token2wav.code2wav_bigvgan_model.activation_post.downsample.conv.weight.requires_grad = False

        d0 = self.token2wav.code2wav_bigvgan_model.activation_post.upsample.conv_transpose.weight.data.shape[0]
        self.token2wav.code2wav_bigvgan_model.activation_post.upsample.conv_transpose.weight.data.copy_(filter.expand(d0, -1, -1))
        self.token2wav.code2wav_bigvgan_model.activation_post.upsample.conv_transpose.weight.requires_grad = False

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.tensor] = None,
        spk: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            spk (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if spk not in self.speaker_map:
            raise ValueError(f"{spk} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value
        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value
        speaker_params = self.speaker_map[spk]

        # 1. Generate from thinker module
        thinker_result = self.thinker.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **thinker_kwargs,
        )
        if not (return_audio and self.has_talker):
            return thinker_result.sequences

        # 2. Generate speech tokens from talker module
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(self.talker.device)
        thinker_token_embeds = [x[0].to(self.talker.device) for x in thinker_result.hidden_states]
        thinker_hidden_states = [x[1][-1].to(self.talker.device) for x in thinker_result.hidden_states]

        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.talker.device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )
        print("input_ids",input_ids.shape)
        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device),
            ],
            dim=1,
        )

        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                self.thinker.get_input_embeddings()(
                    torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device)
                ).to(self.talker.device),
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                self.thinker.get_input_embeddings()(
                    torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device)
                ).to(self.talker.device),
                self.thinker.get_input_embeddings()(
                    torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device)
                ).to(self.talker.device),
            ],
            dim=1,
        )

        talker_attention_mask = torch.cat(
            [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
        ).to(self.talker.device)

        print("talker_input_ids",talker_input_ids.shape)
        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[self.talker.codec_bos_token],
            **{k: (v.to(self.talker.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
        )
        print("talker_result",talker_result.shape)
        talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]
        print("talker_generate_codes",talker_generate_codes.shape)
        # 3. Generate wavs from code
        if self.token2wav.dtype != torch.float:
            self.token2wav.float()
        
        # wav = self.token2wav(
        #     talker_generate_codes.to(self.token2wav.device),
        #     cond=speaker_params["cond"].to(self.token2wav.device).float(),
        #     ref_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
        #     **token2wav_kwargs,
        # )
        effictive_len = talker_generate_codes.shape[1]
        effictive_len = min(effictive_len , self.max_len_talker_generate_codes)
        padded_talker_generate_codes = torch.zeros((1, self.max_len_talker_generate_codes), dtype=talker_generate_codes.dtype, device=self.token2wav.device)
        padded_talker_generate_codes[:, 0:effictive_len] = talker_generate_codes.to(self.token2wav.device)[:, 0:effictive_len]
        wav = self.token2wav(
            padded_talker_generate_codes,
            cond=speaker_params["cond"].to(self.token2wav.device).float(),
            ref_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
            **token2wav_kwargs,
        )
        wav = wav[0:effictive_len*480]
        print("wav",wav.shape)
        return thinker_result.sequences, wav.float()