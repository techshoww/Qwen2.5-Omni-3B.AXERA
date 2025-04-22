from qwen_omni_utils import process_mm_info
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from modeling_export import Qwen2_5OmniModel_Export
import librosa
import audioread
import soundfile as sf
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling
# @title inference function
def inference(video_path):
    messages = [
        {"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
        {"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": 308 * 308, "min_pixels": 308 * 308},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    print("videos",videos[0].shape)
    print('pixel_values_videos', inputs['pixel_values_videos'].shape)
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


device = torch.device("cuda:7")
model_path = "Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)


processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


video_path = "2.mp4"
# video_path="demo.mp4"

## Use a local HuggingFace model to inference.
response, audio  = inference(video_path)
print(response[0])      # 嗯，这钢琴弹得还不错呢。你是在练习曲子吗？还是在即兴创作呀要是有什么关于钢琴的问题，都可以跟我说哦。
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)