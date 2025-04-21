import torch 
from torch import nn
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import onnxruntime as ort
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniModel, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniVisionEncoder, Qwen2_5OmniThinkerConfig,\
Qwen2_5OmniVisionBlock,Qwen2_5OmniVisionSdpaAttention, apply_rotary_pos_emb_vision, Qwen2_5OmniVisionAttention
from audio_export import Qwen2_5OmniAudioEncoder_Export

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
    def __init__(self, config):
        super().__init__(config)

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration_Export(config.thinker_config)