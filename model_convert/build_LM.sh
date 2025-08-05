set -e

INPUT=$1
OUTPUT=$2

pulsar2 llm_build \
                --model_type qwen2_5_omni_text \
                --input_path ${INPUT} \
                --output_path ${OUTPUT} \
                --hidden_state_type bf16 \
                --kv_cache_len 1023 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --chip AX650 


bash tools/embed_process.sh  ${INPUT}   ${OUTPUT}

# cp ${INPUT}*.json ${OUTPUT}
# cp ${INPUT}merges.txt ${OUTPUT}

# chmod 777 ${OUTPUT} -R