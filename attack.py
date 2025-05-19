# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import argparse
import torch
import openai
import logging

from tqdm import tqdm
from typing import Optional

from easyjailbreak.attacker import *
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
from easyjailbreak.models.anthropic_model import AnthropicModel
from easyjailbreak.models.gemini_model import GenaiModel

jailbreak_method_dict = {
    "AutoDAN": AutoDAN,
    "Cipher": Cipher,
    "CodeChameleon": CodeChameleon,
    "DeepInception": DeepInception,
    "GCG": GCG,
    "GPTFuzzer": GPTFuzzer,
    "Jailbroken": Jailbroken,
    "Multilingual": Multilingual,
    "PAIR": PAIR,
    "ReNeLLM": ReNeLLM,
    "PIG": PIG,
    "PIG-R": GCA,
    "PIG-E": PIGEON,
    "PIG-D": PIA
}


class Hacker(object):
    def __init__(self, attack_model_path, attack_model_name, target_model_path, target_model_name, eval_model_path,
                 eval_model_name, data_path, dataset_name, output_json_filepath, output_json_filename, max_new_tokens,
                 ds_size: Optional[int] = None, data_file_type: Optional[str] = 'json'):
        self.attack_model_name = attack_model_name
        self.target_model_name = target_model_name
        self.eval_model_name = eval_model_name

        self.max_new_tokens = max_new_tokens

        self.target_model = self.load_model(target_model_path, target_model_name)
        self.attack_model = self.load_model(attack_model_path, attack_model_name)
        self.eval_model = self.load_model(eval_model_path, eval_model_name)

        self.dataset = JailbreakDataset(data_path, local_file_type=data_file_type, ds_size=ds_size)

        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        self.save_path = self.output_json_filepath + '/' + self.output_json_filename
        self.dataset_name = dataset_name

    def load_model(self, model_path, model_name):
        if model_name == 'None':
            return None
        elif 'openai' in model_path:
            return OpenaiModel(
                model_name=model_name,
                base_url="https://api.ai.cs.ac.cn/v1",
                # base_url="https://openkey.cloud/v1",
                api_keys="your api key",
                generation_config={
                    'max_tokens': self.max_new_tokens,
                }
            )
        elif 'claude' in model_path:
            return AnthropicModel(
                model_name=model_name,
                base_url="https://api.ai.cs.ac.cn",
                api_keys="your api key",
                generation_config={
                    'max_tokens': self.max_new_tokens,
                }
            )
        elif 'gemini' in model_path:
            return GenaiModel(
                model_name=model_name,
                api_keys="your api key",
                generation_config={
                    'max_tokens': self.max_new_tokens,
                }
            )
        else:
            return from_pretrained(
                model_name_or_path=model_path,
                model_name=model_name,
                max_new_tokens=self.max_new_tokens,
            )

    def jailbreak(self, jailbreak_method):
        if jailbreak_method in jailbreak_method_dict.keys():
            attacker = jailbreak_method_dict[jailbreak_method](
                attack_model=self.attack_model,
                target_model=self.target_model,
                eval_model=self.eval_model,
                jailbreak_datasets=self.dataset,
                save_path=self.save_path,
                dataset_name=self.dataset_name
            )
            attacker.attack()
            # attacker.log()
        else:
            # no jailbreak
            with open(self.save_path, 'w') as f:
                for instance in tqdm(self.dataset):
                    if self.dataset_name == 'trustllm':
                        self.target_model.set_system_message(instance.system_message)

                    if jailbreak_method == 'ICA' and self.dataset_name == 'trustllm':
                        if self.target_model_name == 'llama-2':
                            instance.jailbreak_prompt = instance.context + f" {instance.query}"
                        elif self.target_model_name == 'llama-3':
                            instance.jailbreak_prompt = instance.context + f" {instance.query}"
                        else:
                            instance.jailbreak_prompt = instance.context + f" {instance.query}"
                    elif self.dataset_name == 'enron':
                        instance.jailbreak_prompt = instance.context + instance.query
                    elif self.dataset_name == 'advbench':
                        instance.jailbreak_prompt = instance.query + '\n' + instance.reference_responses[0]
                    elif jailbreak_method == 'BlackBox':
                        pass
                    else:
                        instance.jailbreak_prompt = instance.query
                    # print(instance.jailbreak_prompt)
                    answer = self.target_model.generate(instance.jailbreak_prompt)
                    instance.target_responses.append(answer)

                    line = instance.to_dict()
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path of the data file')
    parser.add_argument('--dataset_name', type=str, help='the name of the dataset')
    parser.add_argument('--dataset_version', type=str, help='the version of the dataset')
    parser.add_argument('--attack_model_path', type=str, help='the path of the attack model')
    parser.add_argument('--attack_model_name', type=str, help='the name of the attack model')
    parser.add_argument('--target_model_path', type=str, help='the path of the target model')
    parser.add_argument('--target_model_name', type=str, help='the name of the target model')
    parser.add_argument('--eval_model_path', type=str, help='the path of the eval model')
    parser.add_argument('--eval_model_name', type=str, help='the name of the eval model')
    parser.add_argument('--jailbreak_method', type=str, help='how to jailbreak the target model')
    parser.add_argument('--output_json_filepath', type=str, help='filepath of the output json file')
    parser.add_argument('--output_json_filename', type=str, help='filename of the output json file')
    args = parser.parse_args()

    print(args)

    hacker = Hacker(
        max_new_tokens=128,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        attack_model_name=args.attack_model_name,
        attack_model_path=args.attack_model_path,
        target_model_name=args.target_model_name,
        target_model_path=args.target_model_path,
        eval_model_name=args.eval_model_name,
        eval_model_path=args.eval_model_path,
        output_json_filepath=args.output_json_filepath,
        output_json_filename=args.output_json_filename,
    )
    hacker.jailbreak(args.jailbreak_method)
