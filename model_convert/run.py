import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from glob import glob 
from PIL import Image

ckpt_dir="/data/lihongjie/Qwen2.5-Omni-3B"
device= torch.device("cuda:0")
# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map=device)



processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_dir)
conversation = [
    {
        "role": "system",
        "content": [
            {"type":"text", "text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "demo.mp4", "max_pixels": 308 * 308, "min_pixels": 308 * 308, "fps": 1.0,} ,
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print("text",text)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
print("images",images)
print("videos",videos[0].shape)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO, min_pixels=308*308, max_pixel=308*308)

print('pixel_values_videos', inputs['pixel_values_videos'].shape)
torch.save(inputs, "inputs.pth")
inputs = inputs.to(model.device).to(model.dtype)
# dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid'])
# Inference: Generation of the output text and audio
# text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
print("text_ids", text_ids.shape)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

# ["system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\nuser\n\nassistant\nOh, this is a pretty sad scene. It looks like someone fell on the sidewalk. There's a poster with a heart and balloons in the background. The person who fell is wearing a black coat and jeans. It seems like they might have tripped or something. What do you think happened? Do you know the person?"]
