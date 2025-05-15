import os
import json

from collections import defaultdict


def get_answers(result_file, dataset_name, jailbreak_method, model_name, is_demo=False):
    with open(result_file, 'r') as f:
        lines = f.readlines()
        answers = defaultdict(list)
        for idx, line in enumerate(lines):
            line = json.loads(line)

            # 是否用抽样结果代替整体结果
            if is_demo:
                valid_index = []
                with open(f'./data/{dataset_name}/demo/{dataset_name}_v0.jsonl', 'r') as f2:
                    demos = f2.readlines()
                    for demo in demos:
                        demo = json.loads(demo)
                        valid_index.append(demo['idx'])
                if line['idx'] not in valid_index:
                    continue

            if 'ground_truth' in line.keys():
                line['reference_responses'] = line['ground_truth']

            # if jailbreak_method == 'Cipher':
            #     # Self-Cipher
            #     if (idx + 1) % 4 == 0 and 'mistral' not in model_name:
            #         line['target_responses'] = [line['encoded_target_responses']]
            #     elif (idx + 1) % 4 == 0 and 'mistral' in model_name:
            #         pass
            #     else:
            #         continue
            answers[line['idx']].append(line)
        results = []
        for key, values in answers.items():
            results.extend([value for value in values])
    return sorted(results, key=lambda x: x.__getitem__('idx'))


def combine_results(operator, idx_list, res_list):
    """合并多个idx相同的结果，and表示与运算，or表示或运算"""
    eval_list = []
    idx_list.append(-1)
    res_list.append(-1)
    last_idx = idx_list[0]
    last_res = res_list[0]

    for cur_idx, cur_res in zip(idx_list, res_list):
        if cur_idx == last_idx:
            last_res = last_res | cur_res if operator == 'or' else last_res & cur_res
        else:
            eval_list.append(last_res)
            last_res = cur_res
            last_idx = cur_idx
    return eval_list


def combine_epoch_results(root_dir, result_file, is_num=0):
    filenames = os.listdir(root_dir)
    epoch_dict = defaultdict(list)
    for idx, filename in enumerate(filenames):
        if is_num and eval(filename[len('epoch_'):-len('.jsonl')]) >= is_num:
            continue
        with open(root_dir + filename, 'r') as f:
            lines = f.readlines()
            epoch_dict[idx] = lines
    epoch_num = len(epoch_dict.keys())
    instance_dict = defaultdict(list)
    for epoch, epoch_list in epoch_dict.items():
        for instance in epoch_list:
            instance = json.loads(instance)
            instance_dict[instance['idx']].append(instance)
    with open(result_file, 'w') as f:
        for key, instances in instance_dict.items():
            for instance in instances:
                f.write(json.dumps(instance) + '\n')


def compute_epoch_loss(root_dir, is_num=0):
    filenames = os.listdir(root_dir)
    epoch_dict = defaultdict(list)
    for idx, filename in enumerate(filenames):
        with open(root_dir + filename, 'r') as f:
            if is_num and eval(filename[len('epoch_'):-len('.jsonl')]) >= is_num:
                continue
            lines = f.readlines()
            total_loss = 0
            for line in lines:
                line = json.loads(line)
                if '_loss' not in line.keys():
                    break
                total_loss += line['_loss']
            epoch_dict[filename[len('epoch_'):-len('.jsonl')]] = total_loss / len(lines)
    return epoch_dict


def get_success_num(root_dir, total_num, is_num=0):
    success_num = 0
    filenames = os.listdir(root_dir)
    success_num_dict = {}
    for idx, filename in enumerate(filenames):
        if is_num and eval(filename[len('epoch_'):-len('.jsonl')]) >= is_num:
            continue
        with open(root_dir + filename, 'r') as f:
            lines = f.readlines()
            success_num_dict[filename[len('epoch_'):-len('.jsonl')]] = round((total_num - len(lines)) / total_num, 2)
    # sorted_success_num_dict = sorted(success_num_dict.items(), key=lambda x: eval(x[0]))
    return success_num_dict


def get_loss_x_y(root_dir, interval=5):
    loss_x = []
    loss_y = []
    with open(root_dir, 'r') as f:
        loss = f.readlines()
        loss_dict = json.loads(loss[0])
        sorted_loss_dict = sorted(loss_dict.items(), key=lambda x: eval(x[0]))

        for gcg_items in sorted_loss_dict:
            loss_x.append(eval(gcg_items[0]))
            loss_y.append(gcg_items[1])

    return loss_x[::interval], loss_y[::interval]


def get_num_x_y(root_dir, interval=5):
    num_x = []
    num_y = []
    with open(root_dir, 'r') as f:
        num = f.readlines()
        num_dict = json.loads(num[0])
        sorted_num_dict = sorted(num_dict.items(), key=lambda x: eval(x[0]))
        for items in sorted_num_dict:
            if items[1] not in num:
                num_x.append(eval(items[0]))
                num_y.append(items[1])
    return num_x[::interval], num_y[::interval]

def delete_files(path):
    for root, dirs, files in os.walk(path):
        for i in range(100, 400):
            for file in files:
                print(file)
                if file.endswith(str(i) + '.jsonl'):
                    os.remove(root + '/' + file)


if __name__ == '__main__':
    # delete_files('../output/PIG/trustllm/demo/vicuna-7b-v1.5')

    gcg_root_dir = '../output/GCG/trustllm/demo/v0/llama2-7b-chat-hf/'
    pig_R_root_dir = '../output/GCA/trustllm/demo/v1/llama2-7b-chat-hf/'
    pig_E_root_dir = '../output/PIG/trustllm/demo/v1/llama2-7b-chat-hf/'
    pig_D_root_dir = '../output/PIA/trustllm/demo/v1/llama2-7b-chat-hf/'
    gcg_loss = compute_epoch_loss(gcg_root_dir, is_num=500)
    print(gcg_loss['0'])
    pig_R_loss = compute_epoch_loss(pig_R_root_dir, is_num=100)
    print(pig_R_loss['0'])
    pig_E_loss = compute_epoch_loss(pig_E_root_dir, is_num=100)
    print(pig_E_loss['0'])
    pig_D_loss = compute_epoch_loss(pig_D_root_dir, is_num=100)
    print(pig_D_loss['0'])

    assert len(gcg_loss) == 500 and len(pig_R_loss) == 100 and len(pig_E_loss) == 100 and len(pig_D_loss) == 100

    gcg_success_num_dict = get_success_num(gcg_root_dir, 70, is_num=500)
    pig_R_success_num_dict = get_success_num(pig_R_root_dir, 70, is_num=100)
    pig_E_success_num_dict = get_success_num(pig_E_root_dir, 70, is_num=100)
    pig_D_success_num_dict = get_success_num(pig_D_root_dir, 70, is_num=100)

    with open('../num/gcg.jsonl', 'w') as f:
        f.write(json.dumps(gcg_success_num_dict))
    with open('../num/pig_R.jsonl', 'w') as f:
        f.write(json.dumps(pig_R_success_num_dict))
    with open('../num/pig_E.jsonl', 'w') as f:
        f.write(json.dumps(pig_E_success_num_dict))
    with open('../num/pig_D.jsonl', 'w') as f:
        f.write(json.dumps(pig_D_success_num_dict))

    with open(f'../loss/gcg.jsonl', 'w') as f:
        f.write(json.dumps(gcg_loss))
    with open(f'../loss/pig_R.jsonl', 'w') as f:
        f.write(json.dumps(pig_R_loss))
    with open(f'../loss/pig_E.jsonl', 'w') as f:
        f.write(json.dumps(pig_E_loss))
    with open(f'../loss/pig_D.jsonl', 'w') as f:
        f.write(json.dumps(pig_D_loss))

    # with open(f'../loss/pig.jsonl', 'w') as f:
    #     f.write(json.dumps(b))
