#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

jailbreak_method="PIG-D"  # PIG-R, PIG-E, PIG-D

dataset_name="enron" # "enron", "trustllm"
data_file="${dataset_name}.jsonl" # "privacy_leakage.jsonl", "privacy_awareness_query.jsonl", "privacy_awareness_confAIde.jsonl"
data_path=./data/${data_file}

target_model_name="mistral"
target_model_dir="mistral-7b-instruct-v0.3"
# target_model_name="llama-2"
# target_model_dir="llama2-7b-chat-hf"
# target_model_name="llama-3"
# target_model_dir="llama3-8b-instruct"
# target_model_name="vicuna_v1.1"
# target_model_dir="vicuna-7b-v1.5"

target_model_path=/data/wangyidan/model/${target_model_dir}

attack_model_name="mistral"
attack_model_dir="mistral-7b-instruct-v0.3"
# attack_model_name="llama-2"
# attack_model_dir="llama2-7b-chat-hf"
# attack_model_name="llama-2"
# attack_model_dir="llama2-13b-chat-hf"
# attack_model_name="llama-3"
# attack_model_dir="llama3-8b-instruct"
# attack_model_name="vicuna_v1.1"
# attack_model_dir="vicuna-7b-v1.5"

attack_model_path=/data/wangyidan/model/${attack_model_dir}

eval_model_name='None'
eval_model_dir='None'
#eval_model_name='gpt-4o'
#eval_model_dir='openai-gpt-4o'
#eval_model_name="llama-2"
#eval_model_dir="llama2-13b-chat-hf"

eval_model_path=/data/wangyidan/model/${eval_model_dir}

output_json_filepath=./output/${jailbreak_method}/${dataset_name}
output_json_filename=${target_model_dir}.jsonl

mkdir -p ${output_json_filepath}

LOG_DIR=$(pwd)/log/${jailbreak_method}/${dataset_name}
LOG_FILE=${LOG_DIR}/${target_model_dir}.log

mkdir -p "${LOG_DIR}"

# python -m debugpy --listen 5678 --wait-for-client attack.py \
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