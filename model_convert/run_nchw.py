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

device = torch.device("cuda")
# default: Load the model on the available device(s)
model = Qwen2_5OmniModel_Infer.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.float32, device_map=device)
model.thinker.visual.forward = model.thinker.visual.forward_by_second_nchw

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniModel.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
# paths = sorted(glob("demo_cv308/*.jpg"))
# images = [Image.open(p) for p in paths]
conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "2.mp4", "max_pixels": 308 * 308, "min_pixels": 308 * 308, "fps": 1.0,},
        ],
    },
]

# paths = sorted(glob("demo_cv308/*.jpg"))
# print(paths)
# paths=paths

# images = []
# for p in paths:
#     img = Image.open(p)
#     images.append(img)

img_processor = Qwen2VLImageProcessorExport(max_pixels=308*308, patch_size=14, temporal_patch_size=2, merge_size=2)

image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ]

image_std =  [
    0.26862954,
    0.26130258,
    0.27577711
  ]
# pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
#                                     do_rescale=True, rescale_factor=1/255, do_normalize=True, 
#                                     image_mean=image_mean, image_std=image_std,do_convert_rgb=True)



# set use audio in video
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print("text",text)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
print("images",images)

print("videos",videos[0].shape)

pixel_values, grid_thw = img_processor._preprocess(videos[0], do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)

t,seq_len,tpp,_ = pixel_values.shape

pixel_values = torch.from_numpy(pixel_values).to("cuda")
mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
mean = mean.to("cuda")
std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
std = std.to("cuda")
pixel_values = (pixel_values-mean)/std

pixel_values = pixel_values.permute(0,3,1,2).to("cuda")


inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# inputs["pixel_values_videos"] = pixel_values
print('pixel_values_videos', inputs['pixel_values_videos'].shape)
inputs["pixel_values_videos"] = inputs["pixel_values_videos"].view(2,484,3,392).permute(0,2,1,3)

# np.save("/data/tmp/yongqiang/nfs/lhj/Qwen2.5-Omni-7B-Infer/pixel_values_videos.npy", inputs["pixel_values_videos"].cpu().numpy())
# inputs['input_ids'] = torch.load("input_ids.pth")
# np.save("/data/tmp/yongqiang/nfs/lhj/Qwen2.5-Omni-7B-Infer/input_ids.npy", inputs['input_ids'].cpu().numpy())
torch.save(inputs, "inputs_nchw.pth")
inputs = inputs.to(model.device).to(model.dtype)
# dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid'])
# Inference: Generation of the output text and audio
# text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
# sf.write(
#     "output.wav",
#     audio.reshape(-1).detach().cpu().numpy(),
#     samplerate=24000,
# )

