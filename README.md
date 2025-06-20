# Qwen2.5-VL-3B-Instruct DEMO on Axera
- 视频理解模型转换参考 [模型转换](/model_convert/README.md)
- cpp demo 开发中

## 支持平台

- [x] AX650N


### Python API 运行

#### Requirements

```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
pip install qwen-omni-utils[decord]
pip install soundfile
pip install https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc1/axengine-0.1.3-py3-none-any.whl
```


**视频理解示例**

在开发板上运行命令

```
cd python
python3 run_axinfer.py
```  
**输入**
```
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
```

[视频片段](python/2.mp4)

**文字输出**  
```
It's a Nord Electro 6 keyboard. It's a really popular one. It has a lot of features like different sounds and effects. Have you played with it much?<|im_end|>
```
**语音输出**
[语音](python/output.wav)