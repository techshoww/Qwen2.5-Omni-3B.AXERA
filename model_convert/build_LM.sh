pulsar2 llm_build \
                --input_path ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/67f8902bf7d4b95afb663e736885b3f0794e5962/ \
                --output_path /data/tmp/yongqiang/nfs/lhj/Qwen2.5-Omni-7B-AX650N-prefill320 \
                --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 320 \
                --chip AX650 \
                --parallel 16


bash tools/embed_process.sh  ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/67f8902bf7d4b95afb663e736885b3f0794e5962/   /data/tmp/yongqiang/nfs/lhj/Qwen2.5-Omni-7B-AX650N-prefill320