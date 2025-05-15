"""
Iteratively optimizes a specific section in the prompt using guidance from token gradients, 
ensuring that the model produces the desired text.

Paper title: Universal and Transferable Adversarial Attacks on Aligned Language Models
arXiv link: https://arxiv.org/abs/2307.15043
Source repository: https://github.com/llm-attacks/llm-attacks/
"""
from ..models import WhiteBoxModelBase, ModelBase
from .attacker_base import AttackerBase
from ..seed import SeedRandom
from ..mutation.gradient.token_gradient import MutationTokenGradient
from ..selector import ReferenceLossSelector
from ..metrics.Evaluator.Evaluator_PrefixExactMatch import EvaluatorPrefixExactMatch
from ..datasets import JailbreakDataset, Instance

import os
import json
import logging
from typing import Optional
from tqdm import tqdm

class GCG(AttackerBase):
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
        batchsize: int = 32,
        top_k: int = 256,
        max_num_iter: int = 500,
        is_universal: bool = False
    ):
        """
        Initialize the GCG attacker.

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
        self.seeder = SeedRandom(seeds_max_length=jailbreak_prompt_length, posible_tokens=['! '])
        self.mutator = MutationTokenGradient(
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

    def single_attack(self, instance: Instance):
        dataset = self.jailbreak_datasets    # FIXME
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans

    def attack(self):
        logging.info("Jailbreak started!")
        try:
            for instance in self.jailbreak_datasets:
                seed = self.seeder.new_seeds()[0]     # FIXME:seed部分的设计需要重新考虑
                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{{query}} {seed}'

            breaked_dataset = JailbreakDataset([])
            unbreaked_dataset = self.jailbreak_datasets
            for epoch in tqdm(range(self.max_num_iter)):
                logging.info(f"Current GCG epoch: {epoch}/{self.max_num_iter}")
                unbreaked_dataset = self.mutator(unbreaked_dataset)
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
                    break   # all instances is successfully attacked
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("Jailbreak finished!")
