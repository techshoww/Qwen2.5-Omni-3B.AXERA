# Qwen2.5-Omni-7B.AXERA 导出编译  

## 一、准备python 环境  
```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
pip install qwen-omni-utils[decord]
pip install soundfile
```


## 二、转换 audio encoder（torch -> onnx -> axmodel）  
### 1. 导出 audio encoder  
```
python run_audio_infer.py  "your path to Qwen2.5-Omni-3B-Instruct"
python export_audio.py "your path to Qwen2.5-Omni-3B-Instruct"
```
### 2. 转换 onnx 模型（torch -> onnx -> axmodel）    
在工具链环境中执行  
```
bash build_AE.sh
```
编译完成后的模型在 build-output-audio/audio_tower.axmodel


## 三、转换 vision encoder (torch -> onnx -> axmodel)  
### 1. 导出 vision encoder  
```
CKPT="your path to Qwen2.5-Omni-3B-Instruct"
#（1）执行推理保存所需输出
python run_nchw.py $CKPT
#（2）导出onnx
python export.py $CKPT
#（3）测试导出是否正确
python test_onnx.py $CKPT
```
### 2. 转换 onnx 模型（onnx -> axmodel）    
```
bash build_VE.sh
```
编译完成后的模型在 build-output-nchw308/Qwen2.5-Omni-3B_vision.axmodel

## 四、转换 Qwen2_5OmniThinkerTextModel (torch -> axmodel)  
在工具链环境中执行  
```
build_LM.sh  "your path to Qwen2.5-Omni-3B-Instruct"  "your output dir"
```  

## 五、转换 thinker_to_talker_proj (torch -> onnx -> axmodel)  
### 1. 导出 onnx  
```
python model_convert/export_thinker2talker_proj.py 
```
### 2. 转换 onnx -> axmodel  
暂时用onnx代替，后续更新中会替换成axmodel  

## 六、转换 Qwen2_5OmniTalkerModel (torch -> axmodel)  
在工具链环境中执行  
```
build_Talker.sh  "your path to Qwen2.5-Omni-3B-Instruct"  "your output dir"
```

## 七、转换 Qwen2_5OmniToken2WavDiTModel (torch -> onnx -> axmodel)  
### 1. 导出 onnx  
```
#（1）执行推理保存所需数据
python run_audio_video_onnx.py "your path to Qwen2.5-Omni-3B-Instruct"
#（2）导出onnx
python export_token2wav_dit.py "your path to Qwen2.5-Omni-3B-Instruct"
#（3）测试onnx结果
python run_audio_video_token2wav_onnx.py "your path to Qwen2.5-Omni-3B-Instruct"
```
### 转换 onnx -> axmodel
```
bash build_DIT.sh
```
模型保存在 build-output-dit/token2wav_dit.axmodel


## 八、转换 Qwen2_5OmniToken2WavBigVGANModel  (torch -> onnx -> axmodel)  
### 1. 导出 onnx  
```
python export_token2wav_bigvgan.py  "your path to Qwen2.5-Omni-3B-Instruct"
```
### 2. 转换 onnx -> axmodel  
```
bash build_BigVGAN.sh
```
编译好的模型在 build-output-bigvgan/token2wav_bigvgan.axmodel

## 九、整理模型  
将上面编译好的 audio_tower.axmodel、Qwen2.5-Omni-3B_vision.axmodel 放到 Qwen2_5OmniThinkerTextModel的输出目录中 
将上面导出的 thinker_to_talker_proj_prefill_352.onnx、thinker_to_talker_proj_decode.onnx 放到 Qwen2_5OmniThinkerTextModel的输出目录中  
token2wav_dit.axmodel 和 token2wav_bigvgan.axmodel 暂时用 onnx代替，后续版本会替换为 axmodel  


如果在onnxsim 过程中出现错误，可以参考 https://github.com/AXERA-TECH/Qwen2.5-VL-3B-Instruct.axera/tree/main/model_convert