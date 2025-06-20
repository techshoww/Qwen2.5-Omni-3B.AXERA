import soundfile as sf
from transformers  import Qwen2_5OmniConfig

from modeling_axinfer import Qwen2_5OmniModel_AXInfer

model_path = "../../Qwen2.5-Omni-3B/"
config = Qwen2_5OmniConfig.from_pretrained(model_path)

thinker_dir="../../Qwen2.5-Omni-3B-AX650N-prefill352-0620-s8/"
talker_dir="../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/"
prefill_len=352
lastN=1023

model = Qwen2_5OmniModel_AXInfer(config, thinker_dir, talker_dir, prefill_len, lastN,  run_dynamic=False, lazy_load=True)

video_path = "2.mp4"
messages = [
            {
                "role": "system",
                "content": [
                    {"type":"text", "text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 308 * 308, "min_pixels": 308 * 308, "fps": 1.0,} ,
                ],
            },
        ]

text, audio = model(messages)
print(text)
sf.write(
    "output_onnx.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)