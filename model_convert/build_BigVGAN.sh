pulsar2 build --input token2wav_bigvgan.onnx --config config_token2wav_bigvgan.json --output_dir build-output-bigvgan --output_name token2wav_bigvgan.axmodel --target_hardware AX650 --compiler.check 0



# [Info]These op contains very big range, maybe cause low precision: ['/layers.31/Add_3', '/Transpose_1', '/Unsqueeze', '/avg_pooler/AveragePool', '/Squeeze', '/Transpose_2', 'op_131:onnx.LayerNormalization']



# Network Quantization Finished.
# [Warning]File build-output-bigvgan/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
# [Warning]File build-output-bigvgan/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
# Do quant optimization
# Traceback (most recent call last):
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/common/error.py", line 13, in wrapper
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/command/build.py", line 808, in quant
#     quantize_native_interface(quant_config)
#   File "/home/lihongjie/AI-support/npu-codebase/quant/quantize.py", line 354, in quantize_native_interface
#     float_graph, quantized = export_axmodel(quant_config, quantized)
#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/lihongjie/AI-support/npu-codebase/quant/quantize.py", line 331, in export_axmodel
#     export_ppq_graph(
#   File "/home/lihongjie/AI-support/npu-codebase/quant/ppq/api/interface.py", line 651, in export_ppq_graph
#     exporter.export(file_path=graph_save_to, config_path=config_save_to, graph=graph, **kwargs)
#   File "/home/lihongjie/AI-support/npu-codebase/quant/ppq/parser/axir_exporter.py", line 68, in export
#     ax_exporter.check(graph, quant_config.input_model, file_path)
#   File "/home/lihongjie/AI-support/npu-codebase/quant/ppq/parser/ax_exporter/axir_export_utils.py", line 1074, in check
#     assert var.dtype == expect_type, f"{var.name} But {var.dtype = } vs {expect_type = }"
#            ^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: /Exp_output_0 But var.dtype = <DataType.UINT16: 4> vs expect_type = <DataType.FP32: 1>

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/pulsar2", line 275, in <module>
#     pulsar2()
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/pulsar2", line 271, in pulsar2
#     args.func(args)
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/pulsar2", line 158, in wrapper
#     handler(msg)
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/common/error.py", line 21, in wrapper
#     error_func(e)
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/command/build.py", line 113, in build_error
#     raise e
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/common/error.py", line 13, in wrapper
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/command/build.py", line 539, in build
#     quant_res = quant(ctx)
#                 ^^^^^^^^^^
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/common/error.py", line 21, in wrapper
#     error_func(e)
#   File "/home/lihongjie/AI-support/npu-codebase/yamain/common/error.py", line 73, in error_func
#     raise CodeException(code, e)
# yamain.common.error.CodeException: (<ErrorCode.QuantError: 3>, AssertionError('/Exp_output_0 But var.dtype = <DataType.UINT16: 4> vs expect_type = <DataType.FP32: 1>'))