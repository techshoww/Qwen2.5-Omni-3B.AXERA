export CUDA_VISIBLE_DEVICES="7,0,1,2,3,4,5,6"
# python run1.py

# export CUDA_VISIBLE_DEVICES=7
# python run_audio_video.py
python run_mgpu.py



# pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8