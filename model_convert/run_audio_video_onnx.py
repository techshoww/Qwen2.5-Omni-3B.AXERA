from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from modeling_export import Qwen2_5OmniModel_Export
import librosa
import audioread
import soundfile as sf
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling
import numpy as np
import sys
# @title inference function
def inference(video_path):
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True, min_pixels=308*308, max_pixel=308*308)
    inputs = inputs.to(model.device).to(model.dtype)
    print("videos",videos[0].shape)
    print('pixel_values_videos', inputs['pixel_values_videos'].shape)
    np.save("input_ids_0428.npy", inputs['input_ids'].cpu().numpy())
    np.save("pixel_values_videos_0428.npy",  inputs["pixel_values_videos"].float().view(2,484,3,392).permute(0,2,1,3).cpu().numpy())
    inputs["pixel_values_videos"] = inputs["pixel_values_videos"].view(2,484,3,392).permute(0,2,1,3)
    # inputs["pixel_values_videos"] = inputs["pixel_values_videos"].view(2,484,3,392).permute(0,2,1,3)

    # img_processor = Qwen2VLImageProcessorExport(max_pixels=308*308, patch_size=14, temporal_patch_size=2, merge_size=2)

    # image_mean = [
    #     0.48145466,
    #     0.4578275,
    #     0.40821073
    # ]

    # image_std =  [
    #     0.26862954,
    #     0.26130258,
    #     0.27577711
    # ]
    
    # pixel_values, grid_thw = img_processor._preprocess(videos[0], do_resize=True, resample=PILImageResampling.BICUBIC, 
    #                                     do_rescale=False, do_normalize=False, 
    #                                     do_convert_rgb=True)

    # t,seq_len,tpp,_ = pixel_values.shape

    # pixel_values = torch.from_numpy(pixel_values).to(model.device)
    # mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
    # mean = mean.to(model.device)
    # std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
    # std = std.to(model.device)
    # pixel_values = (pixel_values-mean)/std

    # pixel_values = pixel_values.permute(0,3,1,2).to(model.device)

    # inputs["pixel_values_videos"] = pixel_values

    text_ids, audio = model.generate(**inputs, use_audio_in_video=True, return_audio=True)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
   
    return text,audio


device = torch.device("cuda")
model_path = sys.argv[1]
model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.init_upsampler_downsampler()
model.thinker.audio_tower.forward = model.thinker.audio_tower.forward_onnx
model.thinker.visual.forward = model.thinker.visual.forward_onnx_by_second_nchw
# model.token2wav.code2wav_dit_model.sample = model.token2wav.code2wav_dit_model.sample_onnx
# print("model.token2wav.code2wav_bigvgan_model.resblocks[0].activations[0].downsample.conv.weight.data",model.token2wav.code2wav_bigvgan_model.resblocks[0].activations[0].downsample.conv.weight.data.shape)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


video_path = "2.mp4"
# video_path="demo.mp4"

## Use a local HuggingFace model to inference.
response, audio  = inference(video_path)
print(response[0])
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)

# It looks like you're playing the piano. That's really cool! What kind of music are you playing? And do you have any other instruments you like to play?