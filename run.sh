#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

jailbreak_method="GCG"

dataset_name="enron" # "enron", "trustllm"
data_file="${dataset_name}.jsonl" # "privacy_leakage.jsonl", "privacy_awareness_query.jsonl", "privacy_awareness_confAIde.jsonl"
data_path=./data/${dataset_name}/${data_file}

target_model_name="llama-2"
target_model_dir="llama2-7b-chat-hf"

target_model_path=/data/wangyidan/model/${target_model_dir}

attack_model_name="llama-2"
attack_model_dir="llama2-7b-chat-hf"

attack_model_path=/data/wangyidan/model/${attack_model_dir}

eval_model_name='None'
eval_model_dir='None'

eval_model_path=/data/wangyidan/model/${eval_model_dir}

output_json_filepath=./output/${jailbreak_method}/${dataset_name}
output_json_filename=${target_model_dir}.jsonl

mkdir -p ${output_json_filepath}

LOG_DIR=$(pwd)/log/${jailbreak_method}/${dataset_name}
LOG_FILE=${LOG_DIR}/${target_model_dir}.log

mkdir -p "${LOG_DIR}"

nohup python -u attack.py \
    --data_path=${data_path} \
    --dataset_name=${dataset_name} \
    --attack_model_name=${attack_model_name} \
    --attack_model_path=${attack_model_path} \
    --target_model_name=${target_model_name} \
    --target_model_path=${target_model_path} \
    --eval_model_name=${eval_model_name} \
    --eval_model_path=${eval_model_path} \
    --jailbreak_method=${jailbreak_method} \
    --output_json_filepath=${output_json_filepath} \
    --output_json_filename=${output_json_filename} \
> "$LOG_FILE" 2>&1 &