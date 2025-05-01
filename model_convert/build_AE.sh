pulsar2 build --input audio_tower.onnx --config config_ptq_base.json --output_dir build-output-audio-endtoend-0429-1 --output_name audio_tower.axmodel --target_hardware AX650 --compiler.check 0



# [Info]These op contains very big range, maybe cause low precision: ['/layers.31/Add_3', '/Transpose_1', '/Unsqueeze', '/avg_pooler/AveragePool', '/Squeeze', '/Transpose_2', 'op_131:onnx.LayerNormalization']