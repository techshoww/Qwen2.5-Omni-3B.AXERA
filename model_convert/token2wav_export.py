import os
import math
import onnxruntime as ort 
import torch 
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniToken2WavDiTModel, RungeKutta4ODESolver, \
            Qwen2_5OmniToken2WavModel, Qwen2_5OmniToken2WavConfig, Qwen2_5OmniToken2WavBigVGANModel, Qwen2_5OmniBigVGANConfig, \
                TorchActivation1d, SnakeBeta, kaiser_sinc_filter1d, AMPBlock, DownSample1d, Qwen2_5OmniDiTConfig,\
                    DiTAttention, apply_rotary_pos_emb,  Res2NetBlock, SqueezeExcitationRes2NetBlock, ECAPA_TimeDelayNet, AttentiveStatisticsPooling,\
                        DiTInputEmbedding
from transformers.utils import logging

logger = logging.get_logger(__name__)

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

class DiTAttention_Export(nn.Module):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self._attn_implementation = config._attn_implementation
        self.is_causal = False

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states,  # noised input x
        position_embeddings=None,  # rotary position embedding for x
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE, will be fixed at next release
        cos, sin = position_embeddings
        
        # query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        query_p1, query_p2 = torch.split(query, split_size_or_sections=[1, query.shape[1]-1], dim=1)
        key_p1, key_p2 = torch.split(key, split_size_or_sections=[1, key.shape[1]-1], dim=1)
        query_p1, key_p1 = apply_rotary_pos_emb(query_p1, key_p1, cos, sin)

        query = torch.cat([query_p1,query_p2], 1)
        key = torch.cat([key_p1, key_p2], 1)

        # attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        attention_weights, _ = sdpa_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        # linear proj
        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output



def calculate_same_padding(kernel_size, dilation):
    """
    根据 kernel_size 和 dilation 计算 padding_size，
    使得 Conv1d 输出与输入长度一致（等价于 padding='same'）
    """
    return (dilation * (kernel_size - 1)) // 2

class TDNNBlock_Export(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()

        padding_size = calculate_same_padding(kernel_size, dilation)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            # padding="same",
            # padding_mode="reflect",
            padding=padding_size
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class SEBlock_Export(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            # padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            # padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x

class Res2NetBlock_Export(Res2NetBlock):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__(in_channels, out_channels, scale, kernel_size, dilation)
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
       
        self.blocks = nn.ModuleList(
            [
                TDNNBlock_Export(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )

class AttentiveStatisticsPooling_Export(AttentiveStatisticsPooling):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__(channels, attention_channels)
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            # padding_mode="reflect",
        )

class SERes2NetBlock_Export(SqueezeExcitationRes2NetBlock):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, res2net_scale, se_channels, kernel_size, dilation)
        
        self.tdnn1 = TDNNBlock_Export(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock_Export(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock_Export(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SEBlock_Export(out_channels, se_channels, out_channels)

class ECAPA_TDNN_Export(ECAPA_TimeDelayNet):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    """

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        assert len(config.enc_channels) == len(config.enc_kernel_sizes)
        assert len(config.enc_channels) == len(config.enc_dilations)
       
        self.blocks = nn.ModuleList()
        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock_Export(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SERes2NetBlock_Export(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock_Export(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling_Export(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            # padding_mode="reflect",
        )

class InputEmbedding_Export(DiTInputEmbedding):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        
        self.spk_encoder = ECAPA_TDNN_Export(config)


class Qwen2_5OmniToken2WavDiTModel_Export(Qwen2_5OmniToken2WavDiTModel):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__(config)
        self.input_embed = InputEmbedding_Export(config)

        self.part_num = 1
        self.part_idx = 0
        


        # if os.path.exists("block_diff.pth"):
        #     self.block_diff = torch.load("block_diff.pth").to(self.device)

        if os.path.exists("rope.pth"):
            self.rope = torch.load("rope.pth")
            self.rope = (self.rope[0].to(self.device), self.rope[1].to(self.device))


    def forward_onnx(self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        ):

        print("test Qwen2_5OmniToken2WavDiTModel Onnx -------------------")
        print("x",x.shape)
        print("cond",cond.shape)
        print("spk",spk.shape)
        print("code",code.shape)
        print("time",time)
        session = ort.InferenceSession("token2wav_dit.onnx", providers=["CPUExecutionProvider"])

        inputs = {"x":x.cpu().numpy(), "cond":cond.cpu().numpy(), "spk":spk.cpu().numpy(), "code":code.cpu().numpy(), "time":time.repeat(x.shape[0]).cpu().numpy()}
        out = session.run(["output"], inputs)[0]
        out = torch.from_numpy(out).to(x.device)
        return out 

    def forward(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        drop_audio_conditioning=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        apply_cfg=True,
    ):
        print("-----------------------------------")
        print("x",x.shape)
        print("cond",cond.shape)
        print("spk",spk.shape)
        print("code",code.shape)
        print("time",time)
        torch.save(x, "x.pth")
        torch.save(cond, "cond.pth")
        torch.save(spk, "spk.pth")
        torch.save(code, "code.pth")
        torch.save(time, "time.pth")
        
        # t: conditioning time, c: context (code + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        code_embed = self.text_embed(code, drop_code=False if apply_cfg else drop_code)
        code_embed_uncond = self.text_embed(code, drop_code=True) if apply_cfg else None
        hidden = self.input_embed(
            x,
            spk,
            cond,
            code_embed,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=code_embed_uncond,
            apply_cfg=apply_cfg,
        )

        # rope = self.rotary_embed(x, torch.arange(seq_len, device=x.device).repeat(batch, 1))
        rope = self.rotary_embed(hidden)

        # if os.path.exists("block_diff.pth"):
        #     block_diff = self.block_diff
        # else:
        #     block_diff = self._create_block_diff(hidden)
        #     self.block_diff = block_diff
        #     torch.save(block_diff, "block_diff.pth")
        block_diff = self._create_block_diff(hidden)
        for block in self.transformer_blocks:
            hidden = block(hidden, t, position_embeddings=rope, block_diff=block_diff)

        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)

        return output

    def forward_part1(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        drop_audio_cond=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        cfg=True,
    ):
        print(f"--------run forward_part {1}/{self.part_num}---------------------------")
        print("x",x.shape)
        print("cond",cond.shape)
        print("spk",spk.shape)
        print("code",code.shape)
        print("time",time)
        torch.save(x, "x.pth")
        torch.save(cond, "cond.pth")
        torch.save(spk, "spk.pth")
        torch.save(code, "code.pth")
        torch.save(time, "time.pth")
        
        # t: conditioning time, c: context (code + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        code_embed = self.text_embed(code, drop_code=False if cfg else drop_code)
        code_embed_uncond = self.text_embed(code, drop_code=True) if cfg else None
        hidden = self.input_embed(
            x,
            spk,
            cond,
            code_embed,
            drop_audio_cond=drop_audio_cond,
            code_embed_uncond=code_embed_uncond,
            apply_cfg=cfg,
        )

        # rope = self.rotary_embed(x, torch.arange(seq_len, device=x.device).repeat(batch, 1))
        rope = self.rotary_embed(hidden)
        self.rope = rope
        torch.save(rope, "rope.pth")

        block_diff = self._create_block_diff(hidden)

        num_blocks = len(self.transformer_blocks)
        num_part1 = round(num_blocks/self.part_num)
        
        for block in self.transformer_blocks[0:num_part1]:
            hidden = block(hidden, t, position_embeddings=rope, block_diff=block_diff)

        if self.part_num==1:
            hidden = self.norm_out(hidden, t)
            hidden = self.proj_out(hidden)

        return hidden,  t

    def forward_part1_onnx(self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        ):

        print(f"--------run forward_part {1}/{self.part_num}---------------------------")
        print("x",x.shape)
        print("cond",cond.shape)
        print("spk",spk.shape)
        print("code",code.shape)
        print("time",time)
        session = ort.InferenceSession("token2wav_dit_part1.onnx", providers=["CPUExecutionProvider"])

        inputs = {"x":x.cpu().numpy(), "cond":cond.cpu().numpy(), "spk":spk.cpu().numpy(), "code":code.cpu().numpy(), "time":time.repeat(x.shape[0]).cpu().numpy()}
        hidden, t = session.run(["hidden",  "t"], inputs)
        hidden = torch.from_numpy(hidden).to(x.device)
        t = torch.from_numpy(t).to(x.device)
        return hidden,  t

    def forward_part2(
        self,
        hidden, 
        t,
    ):
        print(f"--------run forward_part {self.part_idx+1}/{self.part_num}---------------------------")
        print("hidden",hidden.shape)
        print("t", t.shape)
        torch.save(hidden, "hidden_part2.pth")
        torch.save(t, "t_part2.pth")

        num_blocks = len(self.transformer_blocks)
        assert self.part_idx > 0
        num_part1 = round(num_blocks/self.part_num) * self.part_idx
        num_part2 = round(num_blocks/self.part_num) * (self.part_idx+1)
        if self.part_idx == self.part_num-1:
            num_part2 = num_blocks
        # hidden = hidden.to(self.device)
        # t = t.to(self.device)
        # self.rope = (self.rope[0].to(self.device), self.rope[1].to(self.device))
        # self.block_diff = self.block_diff.to(self.device)
        

        block_diff = self._create_block_diff(hidden)

        for block in self.transformer_blocks[num_part1:num_part2]:
            hidden = block(hidden, t, position_embeddings=self.rope, block_diff=block_diff)

        if self.part_idx == self.part_num-1:
            hidden = self.norm_out(hidden, t)
            hidden = self.proj_out(hidden)

        return hidden


    def forward_part2_onnx(
        self,
        hidden, 
        t,
    ):
        print(f"--------run forward_part {self.part_idx+1}/{self.part_num}---------------------------")
        session = ort.InferenceSession(f"token2wav_dit_part{self.part_idx+1}.onnx", providers=["CPUExecutionProvider"])

        inputs = {"hidden":hidden.cpu().numpy(),  "t":t.cpu().numpy()}
        out = session.run(["output"], inputs)[0]
        out = torch.from_numpy(out).to(hidden.device)
        return out 

    
    def call_forward_onnx(self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        ):
        return self.forward_onnx(x, cond, spk, code, time)

    def call_forward_2parts_onnx(self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        ):
        assert self.part_num > 1
        hidden, t = self.forward_part1_onnx(x, cond, spk, code, time)
        for i in range(1, self.part_num):
            self.part_idx=i
            hidden = self.forward_part2_onnx(hidden, t)

        return hidden

    def call_forward(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        drop_audio_conditioning=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        apply_cfg=True,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        return self.forward(x, cond, spk, code, time, drop_audio_conditioning, drop_code, apply_cfg=apply_cfg)

    def call_forward_2parts(
        self,
        x,  # nosied input audio
        cond,  # masked cond audio
        spk,  # spk embedding
        code,  # code
        time,  # time step  # noqa: F821 F722
        drop_audio_cond=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        cfg=True,
    ):
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        assert self.part_num > 1
        hidden,  t = self.forward_part1(x, cond, spk, code, time, drop_audio_cond, drop_code, cfg)
        for i in range(1, self.part_num):
            self.part_idx = i
            hidden = self.forward_part2(hidden, t)

        return hidden
        
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
                pred = self.call_forward(x=x, spk=cond, cond=ref_mel, code=code, time=t, drop_audio_conditioning=False, drop_code=False)
                return pred

            out_put = self.call_forward(x=x, code=code, spk=cond, cond=ref_mel, time=t, apply_cfg=True)
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

    @torch.no_grad()
    def sample_2parts(
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
                pred = self.call_forward_2parts(x=x, spk=cond, cond=ref_mel, code=code, time=t, drop_audio_cond=False, drop_code=False)
                return pred

            out_put = self.call_forward_2parts(x=x, code=code, spk=cond, cond=ref_mel, time=t, cfg=True)
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

    @torch.no_grad()
    def sample_onnx(
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
                pred = self.call_forward_onnx(x=x, spk=cond, cond=ref_mel, code=code, time=t)
                return pred

            out_put = self.call_forward_onnx(x=x, code=code, spk=cond, cond=ref_mel, time=t, )
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

    @torch.no_grad()
    def sample_onnx_2parts(
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
                pred = self.call_forward_2parts_onnx(x=x, spk=cond, cond=ref_mel, code=code, time=t)
                return pred

            out_put = self.call_forward_2parts_onnx(x=x, code=code, spk=cond, cond=ref_mel, time=t, )
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

class UpSample1d_Export(nn.Module):
    def __init__(self, in_channels, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        # print("cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size", 0.5 / ratio, 0.6 / ratio, self.kernel_size)
        # print("filter",filter)
        # 使用 nn.ConvTranspose1d 替代 F.conv_transpose1d
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,  # 输入通道数, 
            out_channels=in_channels,  # 输出通道数
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=in_channels,  # 每个通道独立处理
            padding=0,  # 手动处理填充
            bias=False,  # 不需要偏置
        )

        # 冻结卷积层的权重，并加载预计算的滤波器
        self.conv_transpose.weight.data.copy_(filter.expand(in_channels, -1, -1))
        self.conv_transpose.weight.requires_grad = False  # 冻结权重，使其不可训练

        # 使用 ConstantPad1d 替代 F.pad
        self.constant_pad = nn.ConstantPad1d((self.pad, self.pad), value=0)  # 填充值为 0

    def forward(self, x):
        # 对输入进行填充
        # print("UpSample1d_Export before pad x.shape", x.shape, "self.pad",self.pad)
        # x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.constant_pad(x)
        # print("UpSample1d_Export after pad x.shape", x.shape)

        # 使用 nn.ConvTranspose1d 进行转置卷积操作
        x = self.ratio * self.conv_transpose(x)

        # 截断多余的部分
        x = x[..., self.pad_left : -self.pad_right]

        return x

class DownSample1d_Export(nn.Module):
    def __init__(self, in_channels, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        # print("cutoff, half_width, kernel_size", cutoff, half_width, kernel_size)
        # print("filter",filter)
        
        filter = filter.expand(in_channels, -1, -1)
        
        self.register_buffer("filter", filter, persistent=False)

        # 使用 nn.Conv1d 替代 F.conv1d
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=in_channels,
            padding=0,  # 我们手动处理填充
            bias=False,  # 不需要偏置
        ).to(filter.dtype)

       
        # 由于模型后面还要加载权重，这里的初始化是无效的
        # 冻结卷积层的权重，并加载预计算的滤波器
        # self.conv.weight.data.copy_(filter)
        # self.conv.weight.requires_grad = False  # 冻结权重，使其不可训练

        # 使用 ConstantPad1d 替代 F.pad
        self.constant_pad = nn.ConstantPad1d((self.pad_left, self.pad_right), value=0)  # 填充值为 0

    def forward(self, x):
        # assert torch.allclose(self.filter, self.conv.weight.data) 
        # 对输入进行填充
        # print("DownSample1d_Export before pad x.shape", x.shape, "self.pad_left",self.pad_left, "self.pad_right",self.pad_right)
        # x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        x = self.constant_pad(x)
        # print("DownSample1d_Export after pad x.shape", x.shape)
        # 使用 nn.Conv1d 进行卷积操作
        out = self.conv(x)

        return out
    

class TorchActivation1d_Export(TorchActivation1d):
    def __init__(
        self,
        activation,
        in_channels, 
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__(activation, up_ratio, down_ratio, up_kernel_size, down_kernel_size)
        self.upsample = UpSample1d_Export(in_channels, up_ratio, up_kernel_size)
        self.downsample = DownSample1d_Export(in_channels, down_ratio, down_kernel_size)


class SnakeBeta_Export(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        if hidden_states.shape[-1]<2**18:
            return self.do_forward(hidden_states)
        else:
            K = hidden_states.shape[-1]
            y = []
            for i in range(0,K, 2**16):
                xi = hidden_states[..., i : i+2**16]
                yi = self.do_forward(xi)
                y.append(yi)

            y = torch.cat(y, -1)
            assert y.shape[-1]==hidden_states.shape[-1], f'y:{y.shape}, hidden_states:{hidden_states.shape}'

            return y


    def do_forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states
    
class AMPBlock_Export(AMPBlock):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__(channels, kernel_size, dilation)

        self.activations = nn.ModuleList(
            [TorchActivation1d_Export(activation=SnakeBeta(channels), in_channels= channels) for _ in range(self.num_layers)]
        )

class Qwen2_5OmniToken2WavBigVGANModel_Export(Qwen2_5OmniToken2WavBigVGANModel):
    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__(config)

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock_Export(ch, k, d))

        # post conv
        ch = config.upsample_initial_channel // (2**self.num_upsample_layers)
        self.activation_post = TorchActivation1d_Export(activation=SnakeBeta(ch), in_channels=ch)

    # def forward(self, apm_mel):
    #     print("Qwen2_5OmniToken2WavBigVGANModel_Export apm_mel.shape",apm_mel.shape)
    #     torch.save(apm_mel, "apm_mel.pth")
    #     mel_spec = self.apm_to_db(apm_mel)
    #     # pre conv
    #     hidden = self.conv_pre(mel_spec)

    #     for i in range(self.num_upsamples):
    #         # upsampling
    #         for i_up in range(len(self.ups[i])):
    #             hidden = self.ups[i][i_up](hidden)
    #         # AMP blocks
    #         xs = None
    #         for j in range(self.num_kernels):
    #             if xs is None:
    #                 xs = self.resblocks[i * self.num_kernels + j](hidden)
    #             else:
    #                 xs += self.resblocks[i * self.num_kernels + j](hidden)
    #         hidden = xs / self.num_kernels

    #     # post conv
    #     hidden = self.activation_post(hidden)
    #     hidden = self.conv_post(hidden)
    #     audio = torch.clamp(hidden, min=-1.0, max=1.0)  # bound the output to [-1, 1]

    #     return audio.squeeze().cpu()

    def forward(self, mel_spectrogram):
        # print("Qwen2_5OmniToken2WavBigVGANModel_Export apm_mel.shape",mel_spectrogram.shape)
        # torch.save(mel_spectrogram, "apm_mel.pth")
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output
        
        hidden_representation = self.activation_post(hidden_representation)
        
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze().cpu()
    
    def forward_onnx(self, apm_mel):

        print("test Qwen2_5OmniToken2WavBigVGANModel Onnx -------------------")
      
        session = ort.InferenceSession("token2wav_bigvgan.onnx", providers=["CPUExecutionProvider"])

        inputs = {"apm_mel":apm_mel.cpu().numpy()}
        out = session.run(["output"], inputs)[0]
        out = torch.from_numpy(out).to(apm_mel.device)
        return out 

class Qwen2_5OmniToken2WavModel_Export(Qwen2_5OmniToken2WavModel):
    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)

        attn_impl = config._attn_implementation
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen2_5OmniToken2WavModel will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            logger.warning_once(
                "Qwen2_5OmniToken2WavModel does not support eager attention implementation, " "fall back to sdpa"
            )
            attn_impl = "sdpa"
        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel_Export._from_config(
            config.dit_config, attn_implementation=attn_impl
        )

        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel_Export._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )