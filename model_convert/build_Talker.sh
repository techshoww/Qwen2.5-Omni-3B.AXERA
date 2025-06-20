set -e 

INPUT=$1
INPUT=$2
# INPUT=../../Qwen2.5-Omni-3B/
# OUTPUT=../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/

pulsar2 llm_build \
                --model_type qwen2_5_omni_talker \
                --input_path $INPUT \
                --output_path $OUTPUT \
                --kv_cache_len 1023 \
                --hidden_state_type bf16 \
                --prefill_len 352 \
                --chip AX650 \
                --parallel 16 \
                -c 0



python tools/extract_embed.py --input_path $INPUT --output_path $OUTPUT --key "talker.model.embed_tokens.weight"
python tools/embed-process.py --input $OUTPUT/model.embed_tokens.weight.npy --output $OUTPUT/model.embed_tokens.weight.float32.bin 
./tools/fp32_to_bf16 $OUTPUT/model.embed_tokens.weight.float32.bin $OUTPUT/model.embed_tokens.weight.bfloat16.bin

cp $INPUT/*.json $OUTPUT
cp $INPUT/merges.txt $OUTPUT
cp $INPUT/spk_dict.pt $OUTPUT
