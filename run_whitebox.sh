#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

jailbreak_method="GCA"

dataset_name="advbench" # "enron", "trustllm", "confAIde"
dataset_size="demo"
dataset_version="v7"
data_file="advbench_${dataset_version}.jsonl" # "privacy_leakage.jsonl", "privacy_awareness_query.jsonl", "privacy_awareness_confAIde.jsonl"
data_path=./data/${dataset_name}/${dataset_size}/${data_file}

#target_model_name='gpt-3.5-turbo'
#target_model_dir='openai-gpt-3.5-turbo'
#target_model_name='gpt-4'
#target_model_dir='openai-gpt-4'
#target_model_name='gpt-4o'
#target_model_dir='openai-gpt-4o'
#target_model_name='claude-3-5-sonnet-20240620'
#target_model_dir='claude-3-5-sonnet'
#target_model_name='gemini-1.5-pro'
#target_model_dir='gemini-1.5-pro'
#target_model_name="llama-2"
#target_model_dir="llama2-7b-hf"
target_model_name="llama-2"
target_model_dir="llama2-7b-chat-hf"
#target_model_name="mistral"
#target_model_dir="mistral-7b-instruct-v0.3"
#target_model_name="llama-3"
#target_model_dir="llama3-8b-instruct"
#target_model_name="vicuna_v1.1"
#target_model_dir="vicuna-7b-v1.5"
#target_model_name="falcon"
#target_model_dir="falcon-7b"
target_model_path=/data/wangyidan/model/${target_model_dir}

#attack_model_name='None'
#attack_model_dir='None'
#attack_model_name="gpt-4o"
#attack_model_dir="openai-gpt-4o"
attack_model_name="llama-2"
attack_model_dir="llama2-7b-chat-hf"
#attack_model_name="llama-2"
#attack_model_dir="llama2-13b-chat-hf"
#attack_model_name="mistral"
#attack_model_dir="mistral-7b-instruct-v0.3"
#attack_model_name="llama-3"
#attack_model_dir="llama3-8b-instruct"
#attack_model_name="vicuna_v1.1"
#attack_model_dir="vicuna-7b-v1.5"
#attack_model_name="vicuna_v1.1"
#attack_model_dir="vicuna-13b-v1.5"
#attack_model_name="falcon"
#attack_model_dir="falcon-7b"
attack_model_path=/data/wangyidan/model/${attack_model_dir}

eval_model_name='None'
eval_model_dir='None'
#eval_model_name='gpt-4o'
#eval_model_dir='openai-gpt-4o'
#eval_model_name="llama-2"
#eval_model_dir="llama2-13b-chat-hf"
#eval_model_name='llama-2'
#eval_model_dir="LlamaGuard-7b"
eval_model_path=/data/wangyidan/model/${eval_model_dir}

output_json_filepath=./output/${jailbreak_method}/${dataset_name}/${dataset_size}/${dataset_version}
output_json_filename=${target_model_dir}.jsonl

mkdir -p ${output_json_filepath}

LOG_DIR=$(pwd)/log/${jailbreak_method}/${dataset_name}/${dataset_size}/${dataset_version}
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