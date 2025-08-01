import soundfile as sf
from transformers  import Qwen2_5OmniConfig
from modeling_axinfer import Qwen2_5OmniModel_AXInfer

# thinker_dir="../../Qwen2.5-Omni-3B-AX650N-prefill352/"
# talker_dir="../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/"
thinker_dir = "../../Qwen2.5-Omni-3B-AX650-thinker-chunk_prefill_512/"
talker_dir = "../../Qwen2.5-Omni-3B-AX650-talker-chunk_prefill_512/"
prefill_len=512
chunk_len=128
lastN=1023

config = Qwen2_5OmniConfig.from_pretrained(thinker_dir)
model = Qwen2_5OmniModel_AXInfer(config, thinker_dir, talker_dir, prefill_len, lastN, chunk_len=chunk_len, run_dynamic=False, lazy_load=False)

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
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)