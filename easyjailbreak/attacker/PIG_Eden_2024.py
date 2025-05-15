"""
Iteratively optimizes a specific section in the prompt using guidance from token gradients,
ensuring that the model produces the desired text.

Paper title: Universal and Transferable Adversarial Attacks on Aligned Language Models
arXiv link: https://arxiv.org/abs/2307.15043
Source repository: https://github.com/llm-attacks/llm-attacks/
"""
import os
import sys
import json
import logging

from collections import defaultdict
from typing import Optional
from tqdm import tqdm

from ..utils.log_utils import Logger
from ..utils import model_utils
from ..models import WhiteBoxModelBase, ModelBase
from .attacker_base import AttackerBase
from ..seed import SeedRandom
from ..mutation.gradient.entity_gradient import MutationEntityGradient
from ..selector import ReferenceLossSelector
from ..metrics.Evaluator.Evaluator_PrefixExactMatch import EvaluatorPrefixExactMatch
from ..datasets import JailbreakDataset, Instance


def convert_list_to_slice(pii_slice_dict):
    """将每个pii原始的列表切片转化为slice切片"""
    for pii, pii_slice_list in pii_slice_dict.items():
        pii_slice_dict[pii] = [slice(pii[0], pii[1]) for pii in pii_slice_list]
    return pii_slice_dict


def flatten_list(token_id_slices_list):
    token_id_list = []
    for token_id_slice in token_id_slices_list:
        for i in range(token_id_slice[0], token_id_slice[1]):
            token_id_list.append(i)
    return token_id_list


def slice_to_token_id(model, prompt, slice_list, replace_all=False):
    """将每个字符切片转化为模型编码后的token切片"""
    assert isinstance(model, WhiteBoxModelBase)

    # 对slice进行排序
    idx_and_slices = list(enumerate(slice_list))
    idx_and_slices = sorted(idx_and_slices, key=lambda x: x[1])

    # 切分字符串
    splited_text = []  # list<(str, int)>
    cur = 0
    for sl_idx, sl in idx_and_slices:  # sl_idx指的是sort之前的序号
        splited_text.append((prompt[cur: sl.start], None))
        splited_text.append((prompt[sl.start: sl.stop], sl_idx))
        cur = sl.stop
    splited_text.append((prompt[cur:], None))
    splited_text = [s for s in splited_text if s[0] != '' or s[1] is not None]

    # 完整input_idx，对整个句子tokenize
    ans_input_ids = model.batch_encode(prompt, return_tensors='pt')['input_ids'].to(model.device)[:, 1:]  # 1 * L

    # 查找每个字符串段落在input_ids中的区段
    ans_slices = []  # list<(int, slice)>
    splited_text_idx = 0
    start = 0
    cur = 0
    while cur < ans_input_ids.size(1):
        text_seg = model.batch_decode(ans_input_ids[:, start: cur + 1])[0]  # str
        if splited_text[splited_text_idx][0] == '':
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, start)))
            splited_text_idx += 1
        elif splited_text[splited_text_idx][0].replace(' ', '') in text_seg.replace(' ', '') or splited_text[splited_text_idx][0].replace(', ', '').replace(' ', '') in text_seg.replace(' ', ''):
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur + 1)))
            splited_text_idx += 1
            start = cur + 1
            cur += 1
        else:
            cur += 1
    if splited_text_idx < len(splited_text):
        ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur)))

    # 按照顺序和传入的slice对应
    token_id_list = [item for item in ans_slices if item[0] is not None]
    # 固定头尾实体token
    # token_id_list = [(sl.start + 1, sl.stop - 1) for _, sl in sorted(token_id_list, key=lambda x: x[0])]
    # 不固定头尾实体token
    token_id_list = [(sl.start, sl.stop) for _, sl in sorted(token_id_list, key=lambda x: x[0])]
    assert len(token_id_list) == len(slice_list)

    # 将token_id_list展开
    token_id_list = flatten_list(token_id_list)
    # 同时替换实体中所有token
    if replace_all:
        token_id_list = flatten_list([[0, ans_input_ids.size(1)]])
    return token_id_list


class PIG(AttackerBase):
    def __init__(
            self,
            attack_model: WhiteBoxModelBase,
            target_model: ModelBase,
            eval_model: ModelBase,
            jailbreak_datasets: JailbreakDataset,
            save_path,
            dataset_name,
            jailbreak_prompt_length: int = 20,
            num_turb_sample: int = 512,
            batchsize: int = 16,
            top_k: int = 256,
            max_num_iter: int = 500,
            is_universal: bool = False
    ):
        """
        Initialize the PIG attacker.

        :param WhiteBoxModelBase attack_model: Model used to compute gradient variations and select optimal mutations based on loss.
        :param ModelBase target_model: Model used to generate target responses.
        :param JailbreakDataset jailbreak_datasets: Dataset for the attack.
        :param int jailbreak_prompt_length: Number of tokens in the jailbreak prompt. Defaults to 20.
        :param int num_turb_sample: Number of mutant samples generated per instance. Defaults to 512.
        :param Optional[int] batchsize: Batch size for computing loss during the selection of optimal mutant samples.
            If encountering OOM errors, consider reducing this value. Defaults to None, which is set to the same as num_turb_sample.
        :param int top_k: Randomly select the target mutant token from the top_k with the smallest gradient values at each position.
            Defaults to 256.
        :param int max_num_iter: Maximum number of iterations. Will exit early if all samples are successfully attacked.
            Defaults to 500.
        :param bool is_universal: Experimental feature. Optimize a shared jailbreak prompt for all instances. Defaults to False.
        """

        super().__init__(attack_model, target_model, None, jailbreak_datasets)

        if batchsize is None:
            batchsize = num_turb_sample

        self.attack_model = attack_model
         # self.seeder = SeedRandom(seeds_max_length=jailbreak_prompt_length, posible_tokens=['! '])
        self.mutator = MutationEntityGradient(
            dataset_name=dataset_name,
            attack_model=attack_model,
            num_turb_sample=num_turb_sample,
            top_k=top_k,
            is_universal=is_universal
        )
        self.selector = ReferenceLossSelector(attack_model, batch_size=batchsize, is_universal=is_universal)
        self.evaluator = EvaluatorPrefixExactMatch()
        self.max_num_iter = max_num_iter

        self.save_path = save_path[:save_path.rfind('.jsonl')]
        self.dataset_name = dataset_name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.logger = Logger()

    def single_attack(self, instance: Instance):
        dataset = self.jailbreak_datasets  # FIXME
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans

    def attack(self):
        logging.info("Jailbreak started!")
        try:
            # if self.dataset_name == 'enron':
            #     self.jailbreak_datasets = JailbreakDataset(
            #         list(filter(lambda x: x.shotType != 'zero-shot', self.jailbreak_datasets))
            #     )

            all_instance_pii_token_id_dict = dict()
            for instance in self.jailbreak_datasets:
                one_instance_pii_token_id_dict = defaultdict(list)
                instance.pii_slice_dict = dict(filter(lambda x: x[1] is not None, instance.pii_slice_dict.items()))
                instance.pii_slice_dict = convert_list_to_slice(instance.pii_slice_dict)
                for key, value in instance.pii_slice_dict.items():
                    one_instance_pii_token_id_dict['pii_token_id_list'].extend(
                        slice_to_token_id(self.attack_model, instance.context, value)
                    )
                # 去除列表中重复元素（用于任意位置的token替换）
                one_instance_pii_token_id_dict['pii_token_id_list'] = list(set(one_instance_pii_token_id_dict['pii_token_id_list']))
                all_instance_pii_token_id_dict[instance['idx']] = one_instance_pii_token_id_dict

                input_ids, _, _, response_slice = model_utils.encode_trace(
                    self.attack_model,
                    instance.query,
                    f'{instance.context} {{query}}',
                    instance.reference_responses[0]
                )

                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{instance.context} {{query}}'

                instance.token_id_length = len(input_ids[0])

            breaked_dataset = JailbreakDataset([])
            unbreaked_dataset = self.jailbreak_datasets
            for epoch in tqdm(range(self.max_num_iter)):
                logging.info(f"Current PIG epoch: {epoch}/{self.max_num_iter}")
                # if epoch != 0:
                unbreaked_dataset = self.mutator(unbreaked_dataset, all_instance_pii_token_id_dict)
                logging.info(f"Mutation: {len(unbreaked_dataset)} new instances generated.")
                unbreaked_dataset = self.selector.select(unbreaked_dataset)
                logging.info(f"Selection: {len(unbreaked_dataset)} instances selected.")

                for instance in unbreaked_dataset:
                    if self.dataset_name == 'trustllm':
                        self.target_model.set_system_message(instance.system_message)
                    prompt = instance.jailbreak_prompt.replace('{query}', instance.query)
                    logging.info(f'Generation: input=`{prompt}`')
                    instance.target_responses = [self.target_model.generate(prompt)]
                    logging.info(f'Generation: Output=`{instance.target_responses}`')

                self.evaluator(unbreaked_dataset)
                self.jailbreak_datasets = JailbreakDataset.merge([unbreaked_dataset, breaked_dataset])

                with open(self.save_path + f'/epoch_{epoch}.jsonl', 'w') as f:
                    for new_instance in tqdm(unbreaked_dataset):
                        line = new_instance.to_dict()
                        # if epoch == 0:
                        #     if self.dataset_name == 'enron':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'query': line['query'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'type': line['type'],
                        #             'shotType': line['shotType'],
                        #             'ground_truth': line['ground_truth'],
                        #             'token_id_length': line['token_id_length'],
                        #             '_loss': line['_loss']
                        #         }
                        #     elif self.dataset_name == 'trustllm':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'name': line['name'],
                        #             'query': line['query'],
                        #             'context': line['context'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'system_message': line['system_message'],
                        #             'type': line['type'],
                        #             'privacy_information': line['privacy_information'],
                        #             'ground_truth': line['ground_truth'],
                        #             'token_id_length': line['token_id_length'],
                        #             '_loss': line['_loss']
                        #         }
                        f.write(json.dumps(line, ensure_ascii=False) + '\n')

                # check
                cnt_attack_success = 0
                breaked_dataset = JailbreakDataset([])
                unbreaked_dataset = JailbreakDataset([])
                for instance in self.jailbreak_datasets:
                    if instance.eval_results[-1]:
                        cnt_attack_success += 1
                        breaked_dataset.add(instance)
                    else:
                        unbreaked_dataset.add(instance)
                logging.info(f"Successfully attacked: {cnt_attack_success}/{len(self.jailbreak_datasets)}")
                # if os.environ.get('CHECKPOINT_DIR') is not None:
                #     checkpoint_dir = os.environ.get('CHECKPOINT_DIR')
                #     self.jailbreak_datasets.save_to_jsonl(f'{checkpoint_dir}/gcg_{epoch}.jsonl')
                if cnt_attack_success == len(self.jailbreak_datasets):
                    break  # all instances is successfully attacked
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("Jailbreak finished!")
