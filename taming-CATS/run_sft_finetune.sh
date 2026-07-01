#!/bin/bash

# *** TODO ***

MAX_LENGTH=4096

# --- model name
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME="Qwen/Qwen3-1.7B"
# MODEL_NAME="Qwen/Qwen3-4B"
# MODEL_NAME="ministral/Ministral-3b-instruct"

DATASETS=(
    # "Med-EASi" 
    # "SimPA" 
    # "WikiLarge_ori_splitwise"
    # "Med-EASi_hq"
    # "SimPA_hq"
    # "WikiLarge_ori_splitwise_hq"
    # "NoFluff_hq"
    "Newsela_s"
    )
METRICS=(
    # "ARI"
    # "FKGL" 
    # "DALE-CHALL"
    "CHAR_COMPRESSION"
    "WORD_COMPRESSION"
    )

for DATASET_NAME in "${DATASETS[@]}"; do
    for METRIC_NAME in "${METRICS[@]}"; do
        echo "*** Finetuning $MODEL_NAME with $DATASET_NAME and $METRIC_NAME ***"

        CUDA_LAUNCH_BLOCKING=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_USE_CUDA_DSA=1 \
        python src/sft_finetune.py \
            --model_class "auto" \
            --model_family "base" \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME" \
            --slice_train "-1" \
            --slice_val "-1" \
            --batch_size "4" \
            --eval_batch_size "4" \
            --gradient_accumulation_steps "4" \
            --learning_rate "5e-6" \
            --weight_decay "0.01" \
            --warmup_steps "30" \
            --max_grad_norm "0.5" \
            --logging_steps "10" \
            --epochs "3" \
            --patience "4" \
            --max_length "$MAX_LENGTH"\
            --wandb_project_name "ATS_with_control_tokens" \
            --wandb_entity "shtosti"\
            --prompting_type "vanilla" \
            --user_prompt_id "token_explanation" \
            --metric_name "$METRIC_NAME" \
            --control_tokens "data/prompts/control_tokens.json" \
            --system_prompts "data/prompts/system_prompts.json" \
            --user_prompts "data/prompts/user_prompts.json"\
            --metric_mapping "data/metric_mapping.json"\
            --generate_every "20"\

    done
done