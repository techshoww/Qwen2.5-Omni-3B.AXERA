# Qwen2.5-Omni-7B.AXERA

## 一、转换 audio encoder（torch -> onnx -> axmodel）  
### 1. 导出 audio encoder  
```
python run_audio_infer.py  "your path to Qwen2.5-VL-3B-Instruct"
python export_audio.py "your path to Qwen2.5-VL-3B-Instruct"
```
### 2. 转换 onnx 模型（torch -> onnx -> axmodel）    
在工具链环境中执行  
```
bash build_AE.sh
```
编译完成后的模型在 build-output-audio/audio_tower.axmodel


## 二、转换 vision encoder (torch -> onnx -> axmodel)  
### 1. 导出 vision encoder  
```
CKPT="your path to Qwen2.5-VL-3B-Instruct"
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

## 三、转换 Qwen2_5OmniThinkerTextModel (torch -> axmodel)  
在工具链环境中执行  
```
build_LM.sh  "your path to Qwen2.5-VL-3B-Instruct"  "your output dir"
```  
## 四、转换 Qwen2_5OmniTalkerModel (torch -> axmodel)  
在工具链环境中执行  
```
build_Talker.sh  "your path to Qwen2.5-VL-3B-Instruct"  "your output dir"
```

## 五、转换 Qwen2_5OmniToken2WavDiTModel (torch -> onnx -> axmodel)  
### 1. 
