import soundfile as sf
import torch
from transformers import  Qwen2_5OmniProcessor
from modeling_export import Qwen2_5OmniModel_Infer
from qwen_omni_utils import process_mm_info
from glob import glob 
from PIL import Image
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling
import numpy as np
ckpt_dir="/data/lihongjie/Qwen2.5-Omni-3B"
device = torch.device("cuda:1")
# default: Load the model on the available device(s)
model = Qwen2_5OmniModel_Infer.from_pretrained(ckpt_dir, torch_dtype=torch.float32, device_map=device)
model.thinker.visual.forward = model.thinker.visual.forward_by_second_nchw


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
inputs["pixel_values_videos"] = inputs["pixel_values_videos"].view(2,484,3,392).permute(0,2,1,3) # 2,3,488,392

torch.save(inputs, "inputs_nchw.pth")
inputs = inputs.to(model.device).to(model.dtype)
# dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid'])
# Inference: Generation of the output text and audio
# text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
print("text_ids", text_ids.shape)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
# ['system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\nuser\n\nassistant\n嗯，这看起来像是个监控视频。从画面看，有个老人推着婴儿车，突然就摔倒了。可能是老人推车的时候没注意平衡，或者路上有啥东西绊倒了。不过具体原因还得看现场的情况呢。你怎么看这个视频呀。']