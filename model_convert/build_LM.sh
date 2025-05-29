set -e

pulsar2 llm_build \
                --model_type qwen2_5_omni_text \
                --input_path ../../Qwen2.5-Omni-3B/ \
                --output_path ../../Qwen2.5-Omni-3B-AX650N-prefill544/ \
                --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 544 \
                --chip AX650 \
                --parallel 16


bash tools/embed_process.sh  ../../Qwen2.5-Omni-3B/   ../../Qwen2.5-Omni-3B-AX650N-prefill544/

cp ../../Qwen2.5-Omni-3B/*.json ../../Qwen2.5-Omni-3B-AX650N-prefill544/
cp ../../Qwen2.5-Omni-3B/merges.txt ../../Qwen2.5-Omni-3B-AX650N-prefill544/