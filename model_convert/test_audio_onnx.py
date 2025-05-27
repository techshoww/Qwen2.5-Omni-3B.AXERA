
import sys
from qwen_omni_utils import process_mm_info
import librosa

from io import BytesIO
from urllib.request import urlopen
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from audio_export import Qwen2_5OmniModel_Export
# @title inference function
def inference(audio_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": audio_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text





device = torch.device("cpu")
model_path = "/data/lihongjie/Qwen2.5-Omni-3B"
model = Qwen2_5OmniModel_Export.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map=device,
)
model.thinker.audio_tower.forward = model.thinker.audio_tower.forward_onnx
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


audio_path = "1272-128104-0000.flac"
prompt = "Transcribe the English audio into text without any punctuation marks."

audio = librosa.load(BytesIO(open(audio_path, "rb").read()), sr=16000)[0]

## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

audio_path = "cough.wav"
prompt = "Classify the given human vocal sound in English."

audio = librosa.load(BytesIO(open(audio_path, "rb").read()), sr=16000)[0]


## Use a local HuggingFace model to inference.
response = inference(audio_path, prompt=prompt, sys_prompt="You are a vocal sound classification model.")
print(response[0])