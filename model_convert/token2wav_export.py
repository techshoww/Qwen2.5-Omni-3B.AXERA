import onnxruntime as ort 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniToken2WavDiTModel, ODESolverRK4, \
            Qwen2_5OmniToken2WavModel, Qwen2_5OmniToken2WavConfig, Qwen2_5OmniToken2WavBigVGANModel, Qwen2_5OmniBigVGANConfig, \
                TorchActivation1d, SnakeBeta, kaiser_sinc_filter1d, AMPBlock, DownSample1d
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Qwen2_5OmniToken2WavDiTModel_Export(Qwen2_5OmniToken2WavDiTModel):

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

        inputs = {"x":x.cpu().numpy(), "cond":cond.cpu().numpy(), "spk":spk.cpu().numpy(), "code":code.cpu().numpy(), "time":time.cpu().numpy()}
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
        drop_audio_cond=False,  # cfg for cond audio
        drop_code=False,  # cfg for code
        cfg=True,
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
        
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

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
            cfg=cfg,
        )

        # rope = self.rotary_embed(x, torch.arange(seq_len, device=x.device).repeat(batch, 1))
        rope = self.rotary_embed(hidden)

        block_diff = self._create_block_diff(hidden)

        for block in self.transformer_blocks:
            hidden = block(hidden, t, rope=rope, block_diff=block_diff)

        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)

        return output

    @torch.no_grad()
    def sample_onnx(
        self,
        cond,
        ref_mel,
        code,
        steps=10,
        cfg_strength=0.5,
        sway_sampling_coef=-1.0,
    ):
        y_all = torch.randn([1, 30000, self.mel_dim], dtype=ref_mel.dtype)
        max_duration = code.shape[1] * self.repeats
        y0 = y_all[:, :max_duration].to(code.device)
        batch = ref_mel.shape[0]
        cond = cond.unsqueeze(1).repeat(1, max_duration, 1)
        assert batch == 1, "only support batch size = 1 currently"

        def fn(t, x):
            if cfg_strength < 1e-5:
                pred = self.forward_onnx(x=x, spk=cond, cond=ref_mel, code=code, time=t)
                return pred

            out_put = self.forward_onnx(x=x, code=code, spk=cond, cond=ref_mel, time=t, )
            pred, null_pred = torch.chunk(out_put, 2, dim=0)

            return pred + (pred - null_pred) * cfg_strength

        t_start = 0
        t = torch.linspace(t_start, 1, steps, device=code.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = ODESolverRK4(func=fn, y0=y0)
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

    def forward(self, x):
        # 对输入进行填充
        x = F.pad(x, (self.pad, self.pad), mode="replicate")

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

               

    # def forward(self, x):
    #     _, C, _ = x.shape
    #     # print("self.filter.expand(C, -1, -1)", self.filter.expand(C, -1, -1).shape)
    #     x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
    #     out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

    #     return out

    def forward(self, x):
        # assert torch.allclose(self.filter, self.conv.weight.data) 
        # 对输入进行填充
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")

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
        self.activation_post = TorchActivation1d_Export(activation=SnakeBeta(ch), in_channels=ch)

    def forward(self, apm_mel):
        print("Qwen2_5OmniToken2WavBigVGANModel_Export apm_mel.shape",apm_mel.shape)
        torch.save(apm_mel, "apm_mel.pth")
        mel_spec = self.apm_to_db(apm_mel)
        # pre conv
        hidden = self.conv_pre(mel_spec)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                hidden = self.ups[i][i_up](hidden)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](hidden)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](hidden)
            hidden = xs / self.num_kernels

        # post conv
        hidden = self.activation_post(hidden)
        hidden = self.conv_post(hidden)
        audio = torch.clamp(hidden, min=-1.0, max=1.0)  # bound the output to [-1, 1]

        return audio.squeeze().cpu()

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